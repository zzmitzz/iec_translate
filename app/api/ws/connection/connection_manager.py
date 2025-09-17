import asyncio
import json
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket
from .model.room import Room

class ConnectionManager:
    """
    Manages WebSocket connections and rooms for real-time communication.
    Each room contains multiple WebSocket connections that can receive broadcasts.
    """
    
    def __init__(self):
        self.rooms: Dict[int, 'Room'] = {}
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_rooms: Dict[str, int] = {}
        self._connection_counter = 0
    
    async def connect(self, websocket: WebSocket, room_id: int) -> str:
        """
        Accept a new WebSocket connection and add it to a room.
        
        Args:
            websocket: The WebSocket connection to accept
            room_id: The ID of the room to join
            
        Returns:
            str: The unique connection ID
        """
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = f"conn_{self._connection_counter}"
        self._connection_counter += 1
        
        # Store the connection
        self.active_connections[connection_id] = websocket
        self.connection_rooms[connection_id] = room_id
        
        # Add connection to room
        if room_id not in self.rooms:
            # Create new room if it doesn't exist
            self.rooms[room_id] = Room(room_id=room_id)
        
        self.rooms[room_id].add_connection(connection_id)
        
        print(f"Connection {connection_id} joined room {room_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """
        Remove a WebSocket connection from the manager.
        
        Args:
            connection_id: The ID of the connection to remove
        """
        if connection_id in self.connection_rooms:
            room_id = self.connection_rooms[connection_id]
            
            # Remove from room
            if room_id in self.rooms:
                self.rooms[room_id].remove_connection(connection_id)
                
                # Remove empty rooms
                if not self.rooms[room_id].connections:
                    del self.rooms[room_id]
            
            # Remove connection mappings
            del self.connection_rooms[connection_id]
        
        # Remove from active connections
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        print(f"Connection {connection_id} disconnected")
    
    async def broadcast_to_room(self, room_id: int, message: Any, exclude_connection: Optional[str] = None):
        """
        Broadcast a message to all connections in a specific room.
        
        Args:
            room_id: The ID of the room to broadcast to
            message: The message to broadcast (will be JSON serialized)
            exclude_connection: Optional connection ID to exclude from broadcast
        """
        if room_id not in self.rooms:
            return
        
        room = self.rooms[room_id]
        disconnected_connections = []
        
        for connection_id in room.connections:
            if connection_id == exclude_connection:
                continue
                
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                try:
                    # Serialize message to JSON if it's not already a string
                    if not isinstance(message, str):
                        message_str = json.dumps(message)
                    else:
                        message_str = message
                    
                    await websocket.send_text(message_str)
                except Exception as e:
                    print(f"Error broadcasting to connection {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
            else:
                # Connection is no longer active
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    async def send_personal_message(self, connection_id: str, message: Any):
        """
        Send a personal message to a specific connection.
        
        Args:
            connection_id: The ID of the connection to send to
            message: The message to send (will be JSON serialized)
        """
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                # Serialize message to JSON if it's not already a string
                if not isinstance(message, str):
                    message_str = json.dumps(message)
                else:
                    message_str = message
                
                await websocket.send_text(message_str)
            except Exception as e:
                print(f"Error sending personal message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    def get_room_connections(self, room_id: int) -> List[str]:
        """
        Get all connection IDs in a specific room.
        
        Args:
            room_id: The ID of the room
            
        Returns:
            List of connection IDs in the room
        """
        if room_id in self.rooms:
            return list(self.rooms[room_id].connections)
        return []
    
    def get_connection_room(self, connection_id: str) -> Optional[int]:
        """
        Get the room ID that a connection belongs to.
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            Room ID if found, None otherwise
        """
        return self.connection_rooms.get(connection_id)
    
    def get_room_info(self, room_id: int) -> Optional[Dict]:
        """
        Get information about a room including connection count.
        
        Args:
            room_id: The ID of the room
            
        Returns:
            Dictionary with room information or None if room doesn't exist
        """
        if room_id in self.rooms:
            room = self.rooms[room_id]
            return {
                "room_id": room_id,
                "connection_count": len(room.connections),
                "connections": list(room.connections)
            }
        return None
    
    def get_all_rooms_info(self) -> List[Dict]:
        """
        Get information about all rooms.
        
        Returns:
            List of dictionaries with room information
        """
        return [self.get_room_info(room_id) for room_id in self.rooms.keys()]
    
    def get_total_connections(self) -> int:
        """
        Get the total number of active connections.
        
        Returns:
            Total number of active connections
        """
        return len(self.active_connections)


# Global instance of the connection manager
connection_manager = ConnectionManager()
