"""Dark Catppuccin-inspired Qt stylesheet."""

STYLE_SHEET = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 11px;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 70px;
}
QPushButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}
QPushButton:hover { background-color: #45475a; }
QPushButton:checked:hover { background-color: #74a0f5; }
QPushButton:disabled {
    background-color: #181825; color: #585b70; border-color: #313244;
}
QDial {
    background-color: #313244;
    border: 1px solid #585b70;
    border-radius: 4px;
}
QLabel { color: #cdd6f4; background: transparent; }
QComboBox {
    background-color: #313244;
    border: 1px solid #585b70;
    border-radius: 3px;
    padding: 3px 8px;
    min-width: 80px;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background-color: #313244; color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    border: 1px solid #585b70;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px; padding: 0 6px;
    color: #89b4fa;
}
QSlider::groove:vertical { background: #313244; width: 6px; border-radius: 3px; }
QSlider::handle:vertical {
    background: #89b4fa; border: 1px solid #cdd6f4;
    width: 14px; height: 14px; margin: -4px 0; border-radius: 7px;
}
QSlider::handle:vertical:hover { background: #b4d0fb; }
QSlider::add-page:vertical { background: #45475a; border-radius: 3px; }
QSlider::sub-page:vertical { background: #89b4fa; border-radius: 3px; }
QProgressBar {
    background-color: #313244; border: 1px solid #45475a;
    border-radius: 3px; text-align: center; height: 14px;
}
QProgressBar::chunk { background-color: #89b4fa; border-radius: 2px; }
QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }
QTabBar::tab {
    background: #313244; color: #cdd6f4;
    padding: 6px 12px; border: 1px solid #45475a;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #89b4fa; color: #1e1e2e;
    border-bottom: none;
}
"""
