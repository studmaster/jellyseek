from dataclasses import dataclass
from typing import Dict, Callable, Any, Optional
from jellyseek.jellyfin_export.main import fetch_items, save_items
from jellyseek.rag.db_generator import generate_database

@dataclass
class Command:
    """Represents a chat command"""
    name: str
    handler: Callable[..., Any]
    description: str

class CommandHandler:
    def __init__(self):
        self.commands: Dict[str, Command] = {}

    def register(self, name: str, handler: Callable[..., Any], description: str):
        """Register a new command"""
        self.commands[name] = Command(name, handler, description)

    def handle(self, command: str, **kwargs) -> Optional[bool]:
        """Handle a command. Returns True if handled, False if not"""
        cmd = self.commands.get(command)
        if cmd:
            return cmd.handler(**kwargs)
        return None

    def get_help(self) -> str:
        """Get help text for all commands"""
        return "\n".join(
            f"{cmd.name}: {cmd.description}"
            for cmd in self.commands.values()
        )

def cmd_quit(**kwargs) -> bool:
    """Quit the application"""
    return True

def cmd_update(collection, embedding, collection_name, chroma_client, **kwargs) -> bool:
    """Update the movie database"""
    print("\nChecking for updates...")
    
    # Fetch new items from Jellyfin
    new_items = fetch_items()
    if not new_items or not isinstance(new_items, dict) or 'Items' not in new_items:
        print("Failed to fetch valid items from Jellyfin")
        return False
        
    if not new_items['Items']:
        print("No movies found in Jellyfin")
        return False

    try:
        save_items(new_items)
        print("Saved new items, regenerating database...")
        generate_database(force_update=True)
        return True
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        return False

def cmd_help(handler: CommandHandler, **kwargs) -> bool:
    """Show help text"""
    print("\nAvailable Commands:")
    print(handler.get_help())
    return False

def create_command_handler() -> CommandHandler:
    """Create and configure the command handler"""
    handler = CommandHandler()
    
    # Register commands
    handler.register("/quit", cmd_quit, "Exit the application")
    handler.register("/update", cmd_update, "Check for new movies and update the database")
    handler.register("/help", lambda **kwargs: cmd_help(handler, **kwargs), "Show this help message")
    
    return handler