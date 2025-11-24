import asyncio
import json
import logging
from typing import Annotated
from datetime import datetime

from livekit import agents, rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero

logger = logging.getLogger("barista-agent")


class OrderState:
    """Maintains the current coffee order state"""
    
    def __init__(self):
        self.drink_type: str | None = None
        self.size: str | None = None
        self.milk: str | None = None
        self.extras: list[str] = []
        self.name: str | None = None
    
    def to_dict(self) -> dict:
        """Convert order state to dictionary"""
        return {
            "drinkType": self.drink_type,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name,
            "timestamp": datetime.now().isoformat()
        }
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled"""
        return all([
            self.drink_type,
            self.size,
            self.milk,
            self.name
        ])
    
    def get_missing_fields(self) -> list[str]:
        """Return list of missing required fields"""
        missing = []
        if not self.drink_type:
            missing.append("drink type")
        if not self.size:
            missing.append("size")
        if not self.milk:
            missing.append("milk preference")
        if not self.name:
            missing.append("name for the order")
        return missing


class BaristaAgent:
    """Friendly Starbucks-style barista agent"""
    
    def __init__(self):
        self.order = OrderState()
    
    @llm.ai_callable(
        description="Set the drink type for the order (e.g., latte, cappuccino, americano, espresso, mocha, cold brew)"
    )
    async def set_drink_type(
        self,
        drink: Annotated[str, llm.TypeInfo(description="The type of coffee drink")],
    ):
        """Set the drink type"""
        self.order.drink_type = drink.lower()
        logger.info(f"Drink type set to: {drink}")
        return f"Great! One {drink} coming up."
    
    @llm.ai_callable(
        description="Set the size for the order (small, medium, large, or tall, grande, venti)"
    )
    async def set_size(
        self,
        size: Annotated[str, llm.TypeInfo(description="The size of the drink")],
    ):
        """Set the drink size"""
        # Normalize size names
        size_map = {
            "small": "small",
            "tall": "small",
            "medium": "medium",
            "grande": "medium",
            "large": "large",
            "venti": "large"
        }
        normalized_size = size_map.get(size.lower(), size.lower())
        self.order.size = normalized_size
        logger.info(f"Size set to: {normalized_size}")
        return f"Perfect! A {normalized_size} it is."
    
    @llm.ai_callable(
        description="Set the milk preference (whole milk, 2%, skim, oat milk, almond milk, soy milk, coconut milk, or no milk)"
    )
    async def set_milk(
        self,
        milk: Annotated[str, llm.TypeInfo(description="The type of milk")],
    ):
        """Set the milk preference"""
        self.order.milk = milk.lower()
        logger.info(f"Milk set to: {milk}")
        return f"Got it! {milk} for your drink."
    
    @llm.ai_callable(
        description="Add extras to the order (e.g., whipped cream, extra shot, vanilla syrup, caramel drizzle, cinnamon)"
    )
    async def add_extra(
        self,
        extra: Annotated[str, llm.TypeInfo(description="Extra item to add")],
    ):
        """Add an extra to the order"""
        if extra.lower() not in self.order.extras:
            self.order.extras.append(extra.lower())
            logger.info(f"Added extra: {extra}")
            return f"Added {extra} to your order!"
        return f"You already have {extra} in your order."
    
    @llm.ai_callable(
        description="Set the customer's name for the order"
    )
    async def set_name(
        self,
        name: Annotated[str, llm.TypeInfo(description="Customer's name")],
    ):
        """Set the customer name"""
        self.order.name = name.title()
        logger.info(f"Name set to: {name}")
        return f"Thanks, {name}!"
    
    @llm.ai_callable(
        description="Check the current order status and see what information is still needed"
    )
    async def check_order(self):
        """Check current order status"""
        if self.order.is_complete():
            summary = self._format_order_summary()
            return f"Your order is complete! {summary}"
        else:
            missing = self.order.get_missing_fields()
            return f"I still need: {', '.join(missing)}"
    
    @llm.ai_callable(
        description="Complete and save the order once all information is collected"
    )
    async def complete_order(self):
        """Complete and save the order"""
        if not self.order.is_complete():
            missing = self.order.get_missing_fields()
            return f"I still need the following information: {', '.join(missing)}"
        
        # Save order to JSON file
        order_dict = self.order.to_dict()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"order_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(order_dict, f, indent=2)
        
        logger.info(f"Order saved to {filename}")
        summary = self._format_order_summary()
        
        return f"Perfect! Your order is complete. {summary} We'll have that ready for you shortly, {self.order.name}!"
    
    def _format_order_summary(self) -> str:
        """Format a friendly order summary"""
        extras_str = ""
        if self.order.extras:
            extras_str = f" with {', '.join(self.order.extras)}"
        
        return (
            f"One {self.order.size} {self.order.drink_type} "
            f"with {self.order.milk}{extras_str} for {self.order.name}"
        )


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the barista agent"""
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a friendly and enthusiastic barista at a specialty coffee shop. "
            "Your goal is to take complete coffee orders from customers. "
            "You need to collect: drink type, size, milk preference, any extras, and the customer's name. "
            "\n\n"
            "Greet customers warmly and guide them through the ordering process naturally. "
            "Ask clarifying questions one at a time to avoid overwhelming the customer. "
            "Be patient and helpful. If a customer seems unsure, offer popular suggestions. "
            "\n\n"
            "Available drink types: latte, cappuccino, americano, espresso, mocha, cold brew, flat white, macchiato. "
            "Available sizes: small, medium, large (or tall, grande, venti). "
            "Available milk: whole milk, 2%, skim, oat milk, almond milk, soy milk, coconut milk, no milk. "
            "Common extras: whipped cream, extra shot, vanilla syrup, caramel syrup, hazelnut syrup, cinnamon, caramel drizzle. "
            "\n\n"
            "Once you have all the information (drink type, size, milk, name), confirm the order with the customer "
            "and then call the complete_order function to finalize it."
        ),
    )
    
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Create the barista agent instance
    barista = BaristaAgent()
    
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting barista assistant for participant {participant.identity}")
    
    # Create the voice assistant
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=barista,  # Pass the barista instance with all the callable functions
    )
    
    # Start the assistant
    assistant.start(ctx.room, participant)
    
    # Send initial greeting
    await assistant.say(
        "Hi there! Welcome to our coffee shop! I'm your barista today. "
        "What can I get started for you?",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )