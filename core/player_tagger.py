import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime


@dataclass
class Player:
    """Player data container"""
    id: str
    name: str
    number: Optional[int] = None
    position: Optional[str] = None
    team: Optional[str] = None
    headshot_path: Optional[str] = None
    body_crop_path: Optional[str] = None
    created_at: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Player':
        return cls(**data)


class PlayerTagger:
    """
    Manage player identification and tagging.
    Stores player info and crops for future ReID integration.
    """
    
    def __init__(self, database_path: str = "data/headshots"):
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        
        self._players: Dict[str, Player] = {}
        self._db_file = self.database_path / "players.json"
        
        self._load_database()
    
    def _load_database(self):
        """Load player database from JSON"""
        if self._db_file.exists():
            with open(self._db_file, 'r') as f:
                data = json.load(f)
                self._players = {
                    pid: Player.from_dict(pdata) 
                    for pid, pdata in data.items()
                }
    
    def _save_database(self):
        """Save player database to JSON"""
        data = {pid: p.to_dict() for pid, p in self._players.items()}
        with open(self._db_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_player(self, name: str, number: Optional[int] = None,
                   position: Optional[str] = None, team: Optional[str] = None,
                   notes: str = "") -> Player:
        """Add a new player to the database"""
        # Generate unique ID
        player_id = f"player_{len(self._players) + 1:04d}"
        
        player = Player(
            id=player_id,
            name=name,
            number=number,
            position=position,
            team=team,
            notes=notes
        )
        
        self._players[player_id] = player
        self._save_database()
        
        return player
    
    def get_player(self, player_id: str) -> Optional[Player]:
        """Get player by ID"""
        return self._players.get(player_id)
    
    def get_all_players(self) -> List[Player]:
        """Get all players"""
        return list(self._players.values())
    
    def update_player(self, player_id: str, **kwargs) -> Optional[Player]:
        """Update player attributes"""
        if player_id not in self._players:
            return None
        
        player = self._players[player_id]
        for key, value in kwargs.items():
            if hasattr(player, key):
                setattr(player, key, value)
        
        self._save_database()
        return player
    
    def delete_player(self, player_id: str) -> bool:
        """Delete player from database"""
        if player_id not in self._players:
            return False
        
        player = self._players.pop(player_id)
        
        # Delete associated images
        if player.headshot_path and Path(player.headshot_path).exists():
            Path(player.headshot_path).unlink()
        if player.body_crop_path and Path(player.body_crop_path).exists():
            Path(player.body_crop_path).unlink()
        
        self._save_database()
        return True
    
    def save_headshot(self, player_id: str, image: np.ndarray) -> Optional[str]:
        """Save headshot image for player"""
        if player_id not in self._players:
            return None
        
        # Create player directory
        player_dir = self.database_path / player_id
        player_dir.mkdir(exist_ok=True)
        
        # Save image
        headshot_path = player_dir / "headshot.jpg"
        cv2.imwrite(str(headshot_path), image)
        
        # Update player record
        self._players[player_id].headshot_path = str(headshot_path)
        self._save_database()
        
        return str(headshot_path)
    
    def save_body_crop(self, player_id: str, image: np.ndarray) -> Optional[str]:
        """Save body crop image for player (for ReID)"""
        if player_id not in self._players:
            return None
        
        player_dir = self.database_path / player_id
        player_dir.mkdir(exist_ok=True)
        
        # Save with timestamp for multiple crops
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_path = player_dir / f"body_{timestamp}.jpg"
        cv2.imwrite(str(crop_path), image)
        
        # Update player record with latest
        self._players[player_id].body_crop_path = str(crop_path)
        self._save_database()
        
        return str(crop_path)
    
    def get_headshot(self, player_id: str) -> Optional[np.ndarray]:
        """Load headshot image for player"""
        player = self._players.get(player_id)
        if player and player.headshot_path and Path(player.headshot_path).exists():
            return cv2.imread(player.headshot_path)
        return None
    
    def crop_person_from_frame(self, frame: np.ndarray, 
                                bbox: Tuple[int, int, int, int],
                                padding: float = 0.1) -> np.ndarray:
        """Crop person from frame with padding"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return frame[y1:y2, x1:x2].copy()
    
    def search_by_number(self, number: int) -> List[Player]:
        """Search players by jersey number"""
        return [p for p in self._players.values() if p.number == number]
    
    def search_by_name(self, name: str) -> List[Player]:
        """Search players by name (partial match)"""
        name_lower = name.lower()
        return [p for p in self._players.values() if name_lower in p.name.lower()]
    
    def export_for_reid(self) -> Dict[str, List[str]]:
        """
        Export player data structure for ReID training.
        Returns dict mapping player_id to list of image paths.
        """
        export = {}
        
        for player_id, player in self._players.items():
            player_dir = self.database_path / player_id
            if not player_dir.exists():
                continue
            
            # Collect all body crop images
            images = list(player_dir.glob("body_*.jpg"))
            if player.headshot_path and Path(player.headshot_path).exists():
                images.append(Path(player.headshot_path))
            
            if images:
                export[player_id] = [str(p) for p in images]
        
        return export


class LiveTagger:
    """
    Real-time player tagging during video playback.
    Allows clicking on detected persons to assign identity.
    """
    
    def __init__(self, player_tagger: PlayerTagger):
        self.player_tagger = player_tagger
        self._current_assignments: Dict[int, str] = {}  # detection_idx -> player_id
        self._pending_tag: Optional[int] = None  # detection index awaiting tag
    
    def set_pending(self, detection_idx: int):
        """Mark a detection as pending tag assignment"""
        self._pending_tag = detection_idx
    
    def assign_tag(self, player_id: str) -> bool:
        """Assign player to pending detection"""
        if self._pending_tag is None:
            return False
        
        self._current_assignments[self._pending_tag] = player_id
        self._pending_tag = None
        return True
    
    def get_assignment(self, detection_idx: int) -> Optional[str]:
        """Get player ID for detection index"""
        return self._current_assignments.get(detection_idx)
    
    def get_all_assignments(self) -> Dict[int, str]:
        """Get all current assignments"""
        return self._current_assignments.copy()
    
    def clear_assignments(self):
        """Clear all assignments"""
        self._current_assignments.clear()
        self._pending_tag = None
    
    def get_labels(self, num_detections: int) -> List[str]:
        """Get display labels for all detections"""
        labels = []
        for i in range(num_detections):
            player_id = self._current_assignments.get(i)
            if player_id:
                player = self.player_tagger.get_player(player_id)
                if player:
                    label = player.name
                    if player.number:
                        label = f"#{player.number} {label}"
                    labels.append(label)
                else:
                    labels.append(f"ID: {player_id}")
            else:
                labels.append(f"Person {i + 1}")
        return labels
