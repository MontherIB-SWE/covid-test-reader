"""QSS theme built from studio.config tokens."""
from __future__ import annotations

from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    BG_CARD,
    BG_HOVER,
    BG_INPUT,
    BG_PANEL,
    BORDER,
    FG,
    FG_DIM,
    FONT,
    MONO,
    ORANGE,
    ORANGE_HOVER,
    RED,
    RED_HOVER,
)


def application_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {BG};
        color: {FG};
        font-family: "{FONT}";
        font-size: 10pt;
    }}
    QWidget#toolbarPanel {{
        background-color: {BG_PANEL};
        border-bottom: 1px solid {BORDER};
    }}
    QLabel#toolGroupLabel {{
        color: {FG_DIM};
        font-size: 9pt;
        font-weight: 600;
        padding-right: 4px;
    }}
    QFrame#toolbarSeparator {{
        background-color: {BORDER};
        max-width: 1px;
    }}

    /* ── Buttons (default = secondary) ── */
    QPushButton::menu-indicator {{
        image: none;
    }}
    QMenu {{
        background-color: {BG_PANEL};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 0px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 16px;
    }}
    QMenu::item:selected {{
        background-color: {ACCENT};
        color: #ffffff;
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {BORDER};
        margin: 4px 0px;
    }}
    QPushButton {{
        background-color: {BG_CARD};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px 14px;
        min-height: 0px;
        max-height: none;
        font-size: 10pt;
    }}
    QPushButton:hover {{
        background-color: {BG_HOVER};
        border-color: #444444;
    }}
    QPushButton:pressed {{
        background-color: {BG_INPUT};
    }}
    QPushButton:disabled {{
        background-color: {BG_CARD};
        color: {FG_DIM};
        border-color: {BORDER};
    }}
    QPushButton#secondaryButton {{
        background-color: {BG_CARD};
        color: {FG};
    }}
    QPushButton#compactButton {{
        min-width: 36px;
        max-width: 44px;
        padding: 2px 6px;
        font-weight: bold;
    }}

    QPushButton#accentButton {{
        background-color: {ACCENT};
        color: #ffffff;
        border: 1px solid {ACCENT};
        font-weight: 600;
    }}
    QPushButton#accentButton:hover {{
        background-color: {ACCENT_HOVER};
        border-color: {ACCENT_HOVER};
    }}
    QPushButton#accentButton:pressed {{
        background-color: #00945c;
    }}

    QPushButton#dangerButton {{
        background-color: {RED};
        color: #ffffff;
        border: 1px solid {RED};
        font-weight: 600;
    }}
    QPushButton#dangerButton:hover {{
        background-color: {RED_HOVER};
        border-color: {RED_HOVER};
    }}
    QPushButton#dangerButton:pressed {{
        background-color: #c03030;
    }}

    QPushButton#warningButton {{
        background-color: {ORANGE};
        color: #1a1a1a;
        border: 1px solid {ORANGE};
        font-weight: 600;
    }}
    QPushButton#warningButton:hover {{
        background-color: {ORANGE_HOVER};
        border-color: {ORANGE_HOVER};
    }}

    /* Main window tabs */
    QPushButton#tabButton {{
        background-color: {BG_CARD};
        color: {FG};
        border: none;
        border-radius: 6px 6px 0 0;
        padding: 6px 28px;
        min-height: 0px;
        font-size: 11pt;
        font-weight: 600;
    }}
    QPushButton#tabButton:hover {{
        background-color: {BG_HOVER};
    }}
    QPushButton#tabButton:checked {{
        background-color: {BG_PANEL};
        color: {ACCENT};
        border-bottom: 3px solid {ACCENT};
    }}

    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background: {BG};
    }}
    QTabBar::tab {{
        background: {BG_CARD};
        color: {FG};
        padding: 10px 22px;
        margin-right: 2px;
        border-radius: 4px 4px 0 0;
    }}
    QTabBar::tab:selected {{
        background: {BG_PANEL};
        color: {ACCENT};
        font-weight: bold;
    }}
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px 12px 8px 12px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_INPUT};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px 10px;
        min-height: 0px;
        selection-background-color: {ACCENT};
    }}
    QComboBox:hover, QLineEdit:hover {{
        border-color: #444444;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QCheckBox, QRadioButton {{
        spacing: 10px;
        padding: 4px 0;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px;
        height: 18px;
    }}
    QListWidget {{
        background-color: {BG_INPUT};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-family: "{MONO}";
        font-size: 9pt;
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 6px 8px;
        border-radius: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {ACCENT};
        color: #ffffff;
    }}
    QScrollBar:vertical {{
        background: {BG_PANEL};
        width: 12px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        min-height: 28px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {BG_HOVER};
    }}
    QProgressBar {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        text-align: center;
        background: {BG_INPUT};
        min-height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 4px;
    }}
    QLabel#statusLabel {{
        color: {FG_DIM};
        font-family: "{MONO}";
        font-size: 9pt;
    }}
    """
