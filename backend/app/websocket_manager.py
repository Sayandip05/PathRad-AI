"""
WebSocket Manager for Real-Time Status Updates
Emits pipeline state changes to frontend
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Pipeline states
PIPELINE_STATES = [
    "UPLOADED",
    "PREPROCESSING",
    "TRIAGE_RUNNING",
    "RADIOLOGIST_RUNNING",
    "PATHOLOGIST_RUNNING",
    "CLINICAL_CONTEXT_RUNNING",
    "FINALIZING",
    "COMPLETED",
    "ESCALATED_TO_HUMAN",
    "ERROR",
]


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()

        if client_id not in self.active_connections:
            self.active_connections[client_id] = []

        self.active_connections[client_id].append(websocket)
        logger.info(
            f"Client {client_id} connected. Total connections: {len(self.active_connections[client_id])}"
        )

    def disconnect(self, websocket: WebSocket, client_id: str):
        """Remove disconnected client"""
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast_to_patient(self, patient_id: str, message: dict):
        """Broadcast message to all connections for a specific patient"""
        if patient_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[patient_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.active_connections[patient_id].remove(conn)


# Global connection manager
manager = ConnectionManager()


class PipelineStatusEmitter:
    """Emits real-time pipeline status updates"""

    def __init__(self, patient_id: str, diagnosis_id: str):
        self.patient_id = patient_id
        self.diagnosis_id = diagnosis_id

    async def emit_status(
        self,
        stage: str,
        progress: int = 0,
        message: str = "",
        data: Optional[dict] = None,
    ):
        """Emit status update"""
        if stage not in PIPELINE_STATES:
            logger.warning(f"Unknown pipeline state: {stage}")

        status_update = {
            "type": "status_update",
            "patient_id": self.patient_id,
            "diagnosis_id": self.diagnosis_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {},
        }

        logger.info(f"Emitting status: {stage} ({progress}%) - {message}")
        await manager.broadcast_to_patient(self.patient_id, status_update)

    async def emit_agent_result(self, agent_name: str, result: dict):
        """Emit agent completion result"""
        update = {
            "type": "agent_complete",
            "patient_id": self.patient_id,
            "diagnosis_id": self.diagnosis_id,
            "agent": agent_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await manager.broadcast_to_patient(self.patient_id, update)

    async def emit_error(
        self, error_message: str, error_details: Optional[dict] = None
    ):
        """Emit error status"""
        await self.emit_status(
            stage="ERROR",
            progress=0,
            message=error_message,
            data={"error": error_details or {}},
        )

    async def emit_escalation(self, reason: str, details: dict):
        """Emit human review escalation"""
        update = {
            "type": "escalation",
            "patient_id": self.patient_id,
            "diagnosis_id": self.diagnosis_id,
            "reason": reason,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.warning(f"Escalating to human review: {reason}")
        await manager.broadcast_to_patient(self.patient_id, update)

        # Also emit as status
        await self.emit_status(
            stage="ESCALATED_TO_HUMAN",
            progress=100,
            message=f"Escalated to human review: {reason}",
            data=details,
        )


# WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, patient_id)

    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle ping/heartbeat
            if message.get("type") == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, patient_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, patient_id)


# Convenience function for creating emitters
def create_emitter(patient_id: str, diagnosis_id: str) -> PipelineStatusEmitter:
    """Create a status emitter for a diagnosis"""
    return PipelineStatusEmitter(patient_id, diagnosis_id)
