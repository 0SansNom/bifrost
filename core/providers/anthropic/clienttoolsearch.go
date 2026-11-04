package anthropic

import (
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// Client-run tool search on Claude (Codex sends {"type":"tool_search","execution":"client"}).
//
// The caller keeps its deferred tools and runs the search itself, so Anthropic's
// server-side tool_search_tool_* cannot stand in for it: that search only sees tools
// already in the request. Instead the search goes upstream as a plain function, and
// the conversation maps onto Anthropic's custom tool search:
//   - the tool becomes a function named clientToolSearchFunctionName;
//   - a tool_search_call in history becomes a tool_use of that function, and its
//     tool_search_output a tool_result of tool_reference blocks naming the tools the
//     caller loaded (core declares those tools deferred before conversion);
//   - Claude's tool_use of the function goes back to the caller as a client
//     tool_search_call.
//
// This is the reverse of core/providers/openai/toolsearch.go, which runs Claude
// Code's search function as a client tool_search on OpenAI models.

// clientToolSearchFunctionName is the function a client-run tool_search goes upstream as.
const clientToolSearchFunctionName = "tool_search"

// clientToolSearchDescription is used when the caller's tool_search carries none.
const clientToolSearchDescription = "Search for tools to load. Matching tools become available to call."

// anthropicEgressClientToolSearchKeyType keys the function name a client-run
// tool_search went upstream as. It is separate from anthropicClientToolSearchKey, which
// records an Anthropic caller's own search function for the opposite direction.
type anthropicEgressClientToolSearchKeyType struct{}

var anthropicEgressClientToolSearchKey = anthropicEgressClientToolSearchKeyType{}

// clientToolSearchFrom returns the function name recorded for a client-run tool_search
// on this request, or "" when the request declared none.
func clientToolSearchFrom(ctx interface{ Value(any) any }) string {
	if ctx == nil {
		return ""
	}
	name, _ := ctx.Value(anthropicEgressClientToolSearchKey).(string)
	return name
}

// isClientToolSearchTool reports whether tool is a tool_search the caller runs itself.
func isClientToolSearchTool(tool *schemas.ResponsesTool) bool {
	return tool.Type == schemas.ResponsesToolTypeToolSearch && tool.ResponsesToolToolSearch != nil &&
		tool.ResponsesToolToolSearch.Execution != nil && *tool.ResponsesToolToolSearch.Execution == schemas.ToolSearchExecutionClient
}

// hasClientToolSearchTool reports whether tools declare a client-run tool_search.
func hasClientToolSearchTool(tools []schemas.ResponsesTool) bool {
	for i := range tools {
		if isClientToolSearchTool(&tools[i]) {
			return true
		}
	}
	return false
}

// clientToolSearchFunction is the function a client-run tool_search goes upstream as,
// keeping the caller's description and parameters.
func clientToolSearchFunction(tool *schemas.ResponsesTool) *AnthropicTool {
	description := tool.Description
	if description == nil || *description == "" {
		description = schemas.Ptr(clientToolSearchDescription)
	}
	params := tool.ResponsesToolToolSearch.Parameters
	if params == nil {
		params = &schemas.ToolFunctionParameters{
			Type:       "object",
			Properties: schemas.NewOrderedMapFromPairs(schemas.KV("query", map[string]any{"type": "string"})),
			Required:   []string{"query"},
		}
	} else if normalized, err := NormalizeAnthropicToolInputSchema(params); err == nil {
		params = normalized
	}
	return &AnthropicTool{
		Name:        clientToolSearchFunctionName,
		Description: description,
		InputSchema: params,
	}
}

// rewriteClientToolSearchHistory turns the caller's client tool_search_call and
// tool_search_output items into the function_call / function_call_output pair the
// message conversion renders as tool_use / tool_result. The output's tools are named
// by the flat names core declared them under (aliases is the request's
// NamespaceToolAliases). Server-run search items are left for the converter to drop.
// Returns input unchanged, without allocating, when it holds no client search item.
func rewriteClientToolSearchHistory(input []schemas.ResponsesMessage, aliases map[string]schemas.NamespaceToolAlias) []schemas.ResponsesMessage {
	var out []schemas.ResponsesMessage
	var flatName func(namespace, name string) string
	for i := range input {
		msg := &input[i]
		if msg.Type == nil || !msg.IsClientToolSearchItem() || msg.CallID == nil ||
			(*msg.Type != schemas.ResponsesMessageTypeToolSearchCall && *msg.Type != schemas.ResponsesMessageTypeToolSearchOutput) {
			continue
		}
		if out == nil {
			out = make([]schemas.ResponsesMessage, len(input))
			copy(out, input)
			flatName = namespaceFlatNamer(aliases)
		}
		if *msg.Type == schemas.ResponsesMessageTypeToolSearchCall {
			args := "{}"
			if msg.Arguments != nil && *msg.Arguments != "" {
				args = *msg.Arguments
			}
			out[i] = schemas.ResponsesMessage{
				ID:     msg.ID,
				Type:   schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
				Status: msg.Status,
				ResponsesToolMessage: &schemas.ResponsesToolMessage{
					CallID:    msg.CallID,
					Name:      schemas.Ptr(clientToolSearchFunctionName),
					Arguments: &args,
				},
			}
			continue
		}
		var refs []schemas.ResponsesMessageContentBlock
		forEachLoadedTool(msg.ToolSearchOutputTools, func(namespace, name string) {
			refs = append(refs, schemas.ResponsesMessageContentBlock{
				Type:     schemas.ResponsesInputMessageContentBlockTypeToolReference,
				ToolName: schemas.Ptr(flatName(namespace, name)),
			})
		})
		output := &schemas.ResponsesToolMessageOutputStruct{ResponsesFunctionToolCallOutputBlocks: refs}
		if len(refs) == 0 {
			output = &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr("No matching tools found.")}
		}
		out[i] = schemas.ResponsesMessage{
			ID:     msg.ID,
			Type:   schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput),
			Status: msg.Status,
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID: msg.CallID,
				Output: output,
			},
		}
	}
	if out == nil {
		return input
	}
	return out
}

// forEachLoadedTool calls fn for every function a tool_search_output loaded, with its
// namespace ("" for a top-level function).
func forEachLoadedTool(tools []byte, fn func(namespace, name string)) {
	if len(tools) == 0 {
		return
	}
	for _, tool := range gjson.ParseBytes(tools).Array() {
		if tool.Get("type").String() == string(schemas.ResponsesToolTypeNamespace) {
			namespace := tool.Get("name").String()
			for _, member := range tool.Get("tools").Array() {
				if name := member.Get("name").String(); name != "" {
					fn(namespace, name)
				}
			}
			continue
		}
		if name := tool.Get("name").String(); name != "" {
			fn("", name)
		}
	}
}

// namespaceFlatNamer returns the flat name a loaded tool was declared under: its alias
// from the request's flattening, the bare name for a top-level function or a member
// of the default "functions" namespace, else "<namespace>__<name>".
func namespaceFlatNamer(aliases map[string]schemas.NamespaceToolAlias) func(namespace, name string) string {
	byTarget := make(map[schemas.NamespaceToolAlias]string, len(aliases))
	for alias, target := range aliases {
		// Bare-name fallbacks map a function's own name back to it; the qualified alias
		// is the name the tool was declared under.
		if alias != target.Name {
			byTarget[target] = alias
		}
	}
	return func(namespace, name string) string {
		if namespace == "" || namespace == "functions" {
			return name
		}
		if alias, ok := byTarget[schemas.NamespaceToolAlias{Namespace: namespace, Name: name}]; ok {
			return alias
		}
		return namespace + "__" + name
	}
}

// restoreClientToolSearchCall turns Claude's call to the client tool_search function
// back into the client tool_search_call the caller dispatches. It reports whether item
// was rewritten.
func restoreClientToolSearchCall(item *schemas.ResponsesMessage, searchName string) bool {
	if searchName == "" || item == nil || item.Type == nil || *item.Type != schemas.ResponsesMessageTypeFunctionCall ||
		item.ResponsesToolMessage == nil || item.Name == nil || *item.Name != searchName {
		return false
	}
	args := "{}"
	if item.Arguments != nil && *item.Arguments != "" {
		args = *item.Arguments
	}
	item.Type = schemas.Ptr(schemas.ResponsesMessageTypeToolSearchCall)
	item.ResponsesToolMessage = &schemas.ResponsesToolMessage{
		CallID:    item.CallID,
		Arguments: &args,
		Execution: schemas.Ptr(schemas.ToolSearchExecutionClient),
	}
	return true
}

// isStreamedClientToolSearchCall reports whether the stream item at outputIndex is a
// client tool_search_call restored at its content_block_start.
func isStreamedClientToolSearchCall(state *AnthropicResponsesStreamState, outputIndex int) bool {
	item := state.OutputItems[outputIndex]
	return item != nil && item.Type != nil && *item.Type == schemas.ResponsesMessageTypeToolSearchCall &&
		item.ResponsesToolMessage != nil && item.Execution != nil && *item.Execution == schemas.ToolSearchExecutionClient
}
