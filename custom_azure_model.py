"""Custom Azure OpenAI model implementation."""

from typing import List, Dict, Optional
from smolagents import Tool, ChatMessage, AzureOpenAIServerModel
from smolagents.models import parse_json_if_needed

def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    for tool_call in message.tool_calls:
        tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message

class CustomAzureOpenAIServerModel(AzureOpenAIServerModel):
    """Custom Azure OpenAI model that handles stop sequences client-side.
    
    This implementation is specifically designed for models like o4-mini that don't 
    support stop sequences natively. It processes the stop sequences after receiving
    the response from the model.
    """
    
    def _truncate_at_stop_sequence(self, text: str, stop_sequences: List[str]) -> str:
        """Truncate the text at the first occurrence of any stop sequence."""
        if not stop_sequences:
            return text
            
        positions = []
        for stop_seq in stop_sequences:
            pos = text.find(stop_seq)
            if pos != -1:
                positions.append(pos)
                
        if positions:
            earliest_stop = min(positions)
            return text[:earliest_stop]
            
        return text

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        # Remove stop_sequences from kwargs to avoid API errors
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=None,  # Explicitly set to None
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        
        response = self.client.chat.completions.create(**completion_kwargs)
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens

        # Get the response message
        message_dict = response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        
        # Apply stop sequence truncation if needed
        if stop_sequences and "content" in message_dict and message_dict["content"]:
            message_dict["content"] = self._truncate_at_stop_sequence(
                message_dict["content"], 
                stop_sequences
            )

        message = ChatMessage.from_dict(message_dict)
        message.raw = response
        
        if tools_to_call_from is not None:
            return parse_tool_args_if_needed(message)
        return message

# Register the custom model with smolagents
import smolagents
smolagents.CustomAzureOpenAIServerModel = CustomAzureOpenAIServerModel 