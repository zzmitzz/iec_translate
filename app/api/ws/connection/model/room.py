# room.py

from datetime import datetime
from typing import Set, Optional, Dict, List
class Room:
    def __init__(self, room_id: int):
        self.room_id = room_id
        self.connections: Set[str] = set()
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_connection(self, connection_id: str):
        """Add a connection to this room."""
        self.connections.add(connection_id)
        self.last_activity = datetime.now()
    
    def remove_connection(self, connection_id: str):
        """Remove a connection from this room."""
        self.connections.discard(connection_id)
        self.last_activity = datetime.now()
    
    def has_connection(self, connection_id: str) -> bool:
        """Check if a connection exists in this room."""
        return connection_id in self.connections
    
    def get_connection_count(self) -> int:
        """Get the number of connections in this room."""
        return len(self.connections)
    