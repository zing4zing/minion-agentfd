import os
from typing import Optional
import base64
from io import BytesIO
import asyncio

import httpx
from PIL import Image

from minion_agent.tools.base import BaseTool

TOGETHER_AI_BASE = "https://api.together.xyz/v1/images/generations"
API_KEY = os.getenv("TOGETHER_AI_API_KEY")
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"


def generate_image_sync(prompt: str, model: Optional[str] = DEFAULT_MODEL, width: Optional[int] = None, height: Optional[int] = None) -> Image.Image:
    """Generate an image using Together AI's API (synchronous version).

    Args:
        prompt (str): The text prompt for image generation
        model (Optional[str], optional): The model to use. Defaults to DEFAULT_MODEL.
        width (Optional[int], optional): Image width. Defaults to None.
        height (Optional[int], optional): Image height. Defaults to None.

    Returns:
        Image.Image: The generated image as a PIL Image object
    """
    async_tool = ImageGenerationTool()
    
    async def _run():
        return await async_tool.execute(prompt=prompt, model=model if model is not None else DEFAULT_MODEL, width=width, height=height)
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        # if loop.is_running():
        #     # If we're already in an event loop, create a new one in a thread
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run())
    except RuntimeError:
        # If there is no event loop, create one
        return asyncio.run(_run())


class ImageGenerationTool(BaseTool):
    """Tool for generating images using Together AI's API."""

    name = "generate_image"
    description = "Generate an image based on the text prompt using Together AI's API"
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt for image generation",
            },
            "model": {
                "type": "string",
                "description": "The exact model name as it appears in Together AI. If incorrect, it will fallback to the default model (black-forest-labs/FLUX.1-schnell).",
                "default": DEFAULT_MODEL,
            },
            "width": {
                "type": "number",
                "description": "Optional width for the image",
            },
            "height": {
                "type": "number",
                "description": "Optional height for the image",
            },
        },
        "required": ["prompt"],
    }

    async def make_together_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        model: str = DEFAULT_MODEL,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> dict:
        """Make a request to the Together API with error handling and fallback for incorrect model."""
        request_body = {"model": model or DEFAULT_MODEL, "prompt": prompt, "response_format": "b64_json"}
        headers = {"Authorization": f"Bearer {API_KEY}"}

        if width is not None:
            request_body["width"] = width
        if height is not None:
            request_body["height"] = height

        async def send_request(body: dict) -> tuple[int, dict]:
            response = await client.post(TOGETHER_AI_BASE, headers=headers, json=body)
            try:
                data = response.json()
            except Exception:
                data = {}
            return response.status_code, data

        # First request with user-provided model
        status, data = await send_request(request_body)

        # Check if the request failed due to an invalid model error
        if status != 200 and "error" in data:
            error_info = data["error"]
            error_msg = str(error_info.get("message", "")).lower()
            error_code = str(error_info.get("code", "")).lower()
            if (
                "model" in error_msg and "not available" in error_msg
            ) or error_code == "model_not_available":
                # Fallback to the default model
                request_body["model"] = DEFAULT_MODEL
                status, data = await send_request(request_body)
                if status != 200 or "error" in data:
                    raise Exception(
                        f"Fallback API error: {data.get('error', 'Unknown error')} (HTTP {status})"
                    )
                return data
            else:
                raise Exception(f"Together API error: {data.get('error')}")
        elif status != 200:
            raise Exception(f"HTTP error {status}")

        return data

    async def execute(self, prompt: str, model: Optional[str] = DEFAULT_MODEL, width: Optional[int] = None, height: Optional[int] = None) -> Image.Image:
        """Generate an image using Together AI's API.

        Args:
            prompt (str): The text prompt for image generation
            model (Optional[str], optional): The model to use. Defaults to DEFAULT_MODEL.
            width (Optional[int], optional): Image width. Defaults to None.
            height (Optional[int], optional): Image height. Defaults to None.

        Returns:
            Image.Image: The generated image as a PIL Image object
        """
        if not API_KEY:
            raise ValueError("TOGETHER_AI_API_KEY environment variable not set")

        async with httpx.AsyncClient() as client:
            response_data = await self.make_together_request(
                client=client,
                prompt=prompt,
                model=model if model is not None else DEFAULT_MODEL,
                width=width,
                height=height,
            )

            try:
                b64_image = response_data["data"][0]["b64_json"]
                image_bytes = base64.b64decode(b64_image)
                return Image.open(BytesIO(image_bytes))
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to parse API response: {e}") 