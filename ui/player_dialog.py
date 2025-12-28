from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QLineEdit, QSpinBox, QComboBox, QPushButton,
                              QListWidget, QListWidgetItem, QGroupBox, QLabel,
                              QMessageBox, QInputDialog, QWidget, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
from typing import Optional

from core.player_tagger import PlayerTagger, Player


class PlayerListItem(QListWidgetItem):
    """List item with player data"""
    def __init__(self, player: Player):
        display = f"#{player.number} {player.name}" if player.number else player.name
        if player.position:
            display += f" ({player.position})"
        super().__init__(display)
        self.player = player


class PlayerDialog(QDialog):
    """Dialog for managing players and tagging"""
    
    player_selected = pyqtSignal(str)  # Emits player_id when selected
    
    def __init__(self, player_tagger: PlayerTagger, parent=None):
        super().__init__(parent)
        
        self.player_tagger = player_tagger
        self.setWindowTitle("Player Manager")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Splitter for list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # --- Left: Player List ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.player_list = QListWidget()
        self.player_list.itemClicked.connect(self._on_player_selected)
        self.player_list.itemDoubleClicked.connect(self._on_player_double_clicked)
        left_layout.addWidget(self.player_list)
        
        # List buttons
        list_btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Player")
        add_btn.clicked.connect(self._add_player)
        list_btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_player)
        list_btn_layout.addWidget(remove_btn)
        
        left_layout.addLayout(list_btn_layout)
        
        splitter.addWidget(left_widget)
        
        # --- Right: Player Details ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        details_group = QGroupBox("Player Details")
        details_form = QFormLayout(details_group)
        
        self.name_edit = QLineEdit()
        details_form.addRow("Name:", self.name_edit)
        
        self.number_spin = QSpinBox()
        self.number_spin.setRange(0, 99)
        self.number_spin.setSpecialValueText("None")
        details_form.addRow("Number:", self.number_spin)
        
        self.position_combo = QComboBox()
        self.position_combo.addItems([
            "", "GK", "CB", "LB", "RB", "CDM", "CM", "CAM", 
            "LW", "RW", "LM", "RM", "ST", "CF"
        ])
        details_form.addRow("Position:", self.position_combo)
        
        self.team_edit = QLineEdit()
        details_form.addRow("Team:", self.team_edit)
        
        self.notes_edit = QLineEdit()
        details_form.addRow("Notes:", self.notes_edit)
        
        right_layout.addWidget(details_group)
        
        # Headshot preview
        headshot_group = QGroupBox("Headshot")
        headshot_layout = QVBoxLayout(headshot_group)
        
        self.headshot_label = QLabel()
        self.headshot_label.setMinimumSize(150, 150)
        self.headshot_label.setMaximumSize(200, 200)
        self.headshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.headshot_label.setStyleSheet("background-color: #333; border: 1px solid #555;")
        headshot_layout.addWidget(self.headshot_label)
        
        right_layout.addWidget(headshot_group)
        
        # Update button
        update_btn = QPushButton("Update Player")
        update_btn.clicked.connect(self._update_player)
        right_layout.addWidget(update_btn)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_widget)
        splitter.setSizes([250, 350])
        
        # --- Bottom buttons ---
        bottom_layout = QHBoxLayout()
        
        select_btn = QPushButton("Select for Tagging")
        select_btn.clicked.connect(self._select_player)
        bottom_layout.addWidget(select_btn)
        
        bottom_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        
        layout.addLayout(bottom_layout)
        
        # Load players
        self._refresh_list()
        self._selected_player_id: Optional[str] = None
    
    def _refresh_list(self):
        """Refresh the player list"""
        self.player_list.clear()
        
        for player in self.player_tagger.get_all_players():
            item = PlayerListItem(player)
            self.player_list.addItem(item)
    
    def _on_player_selected(self, item: PlayerListItem):
        """Handle player selection"""
        player = item.player
        self._selected_player_id = player.id
        
        # Populate details
        self.name_edit.setText(player.name)
        self.number_spin.setValue(player.number or 0)
        self.position_combo.setCurrentText(player.position or "")
        self.team_edit.setText(player.team or "")
        self.notes_edit.setText(player.notes or "")
        
        # Load headshot
        headshot = self.player_tagger.get_headshot(player.id)
        if headshot is not None:
            self._set_headshot_preview(headshot)
        else:
            self.headshot_label.clear()
            self.headshot_label.setText("No image")
    
    def _on_player_double_clicked(self, item: PlayerListItem):
        """Handle double-click to select for tagging"""
        self._selected_player_id = item.player.id
        self.player_selected.emit(item.player.id)
        self.accept()
    
    def _set_headshot_preview(self, image: np.ndarray):
        """Set headshot preview image"""
        rgb = image[..., ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        scaled = pixmap.scaled(
            self.headshot_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.headshot_label.setPixmap(scaled)
    
    def _add_player(self):
        """Add a new player"""
        name, ok = QInputDialog.getText(self, "Add Player", "Player Name:")
        if ok and name.strip():
            player = self.player_tagger.add_player(name.strip())
            self._refresh_list()
            
            # Select the new player
            for i in range(self.player_list.count()):
                item = self.player_list.item(i)
                if isinstance(item, PlayerListItem) and item.player.id == player.id:
                    self.player_list.setCurrentItem(item)
                    self._on_player_selected(item)
                    break
    
    def _remove_player(self):
        """Remove selected player"""
        if not self._selected_player_id:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            "Are you sure you want to delete this player?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.player_tagger.delete_player(self._selected_player_id)
            self._selected_player_id = None
            self._refresh_list()
            self._clear_details()
    
    def _update_player(self):
        """Update selected player details"""
        if not self._selected_player_id:
            return
        
        number = self.number_spin.value() if self.number_spin.value() > 0 else None
        position = self.position_combo.currentText() or None
        
        self.player_tagger.update_player(
            self._selected_player_id,
            name=self.name_edit.text(),
            number=number,
            position=position,
            team=self.team_edit.text() or None,
            notes=self.notes_edit.text() or ""
        )
        
        self._refresh_list()
    
    def _select_player(self):
        """Emit selected player for tagging"""
        if self._selected_player_id:
            self.player_selected.emit(self._selected_player_id)
            self.accept()
    
    def _clear_details(self):
        """Clear detail fields"""
        self.name_edit.clear()
        self.number_spin.setValue(0)
        self.position_combo.setCurrentIndex(0)
        self.team_edit.clear()
        self.notes_edit.clear()
        self.headshot_label.clear()
    
    def save_headshot_for_current(self, image: np.ndarray) -> bool:
        """Save headshot for currently selected player"""
        if not self._selected_player_id:
            return False
        
        path = self.player_tagger.save_headshot(self._selected_player_id, image)
        if path:
            self._set_headshot_preview(image)
            return True
        return False
    
    def save_body_crop_for_current(self, image: np.ndarray) -> bool:
        """Save body crop for currently selected player"""
        if not self._selected_player_id:
            return False
        
        path = self.player_tagger.save_body_crop(self._selected_player_id, image)
        return path is not None
    
    @property
    def selected_player_id(self) -> Optional[str]:
        return self._selected_player_id


class QuickTagDialog(QDialog):
    """Quick dialog for selecting a player to tag"""
    
    def __init__(self, player_tagger: PlayerTagger, parent=None):
        super().__init__(parent)
        
        self.player_tagger = player_tagger
        self.setWindowTitle("Tag Player")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        
        label = QLabel("Select player to tag:")
        layout.addWidget(label)
        
        self.player_list = QListWidget()
        self.player_list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.player_list)
        
        # Populate
        for player in player_tagger.get_all_players():
            item = PlayerListItem(player)
            self.player_list.addItem(item)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        select_btn = QPushButton("Select")
        select_btn.clicked.connect(self.accept)
        btn_layout.addWidget(select_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self._selected_id: Optional[str] = None
    
    def _on_double_click(self, item: PlayerListItem):
        self._selected_id = item.player.id
        self.accept()
    
    def get_selected_player_id(self) -> Optional[str]:
        item = self.player_list.currentItem()
        if isinstance(item, PlayerListItem):
            return item.player.id
        return self._selected_id
