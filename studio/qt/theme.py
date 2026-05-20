"""QSS theme built from studio.config tokens."""
from __future__ import annotations

from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    BG_ACTIVE,
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
        border-radius: 6px;
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
        padding: 6px 14px;
        min-height: 0px;
        max-height: none;
        font-size: 10pt;
    }}
    QPushButton:hover {{
        background-color: {BG_HOVER};
        border-color: {ACCENT};
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
        padding: 4px 6px;
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
        background-color: #4338ca;
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
        background-color: #b91c1c;
    }}

    QPushButton#warningButton {{
        background-color: {ORANGE};
        color: #090a0f;
        border: 1px solid {ORANGE};
        font-weight: 600;
    }}
    QPushButton#warningButton:hover {{
        background-color: {ORANGE_HOVER};
        border-color: {ORANGE_HOVER};
    }}
    QPushButton#warningButton:pressed {{
        background-color: #b45309;
    }}

    /* ── Main Tab bar ── */
    QWidget#topTabBar {{
        background-color: {BG_PANEL};
        border-bottom: 1px solid {BORDER};
    }}
    QPushButton#tabButton {{
        background-color: transparent;
        color: {FG_DIM};
        border: none;
        border-radius: 6px;
        padding: 6px 20px;
        min-height: 0px;
        font-size: 11pt;
        font-weight: 600;
        margin: 4px;
    }}
    QPushButton#tabButton:hover {{
        background-color: {BG_HOVER};
        color: {FG};
    }}
    QPushButton#tabButton:checked {{
        background-color: {BG_CARD};
        color: {ACCENT};
        border: 1px solid {BORDER};
    }}

    /* ── Inner QTabWidget ── */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        background: {BG};
    }}
    QTabBar::tab {{
        background: {BG_PANEL};
        color: {FG_DIM};
        padding: 8px 20px;
        margin-right: 4px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        border: 1px solid {BORDER};
        border-bottom: none;
    }}
    QTabBar::tab:hover {{
        background: {BG_HOVER};
        color: {FG};
    }}
    QTabBar::tab:selected {{
        background: {BG_CARD};
        color: {ACCENT};
        font-weight: bold;
        border-color: {BORDER};
    }}

    /* ── Group Box (Cards) ── */
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 16px;
        padding: 16px 14px 10px 14px;
        font-weight: 600;
        background-color: {BG_CARD};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px;
        color: {ACCENT};
    }}

    /* ── Inputs ── */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_INPUT};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 6px 12px;
        min-height: 0px;
        selection-background-color: {ACCENT};
    }}
    QComboBox:hover, QLineEdit:hover {{
        border-color: {ACCENT};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ACCENT_HOVER};
        background-color: {BG_ACTIVE};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border: none;
        width: 0;
        height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid {FG_DIM};
        margin-right: 8px;
    }}
    QComboBox::down-arrow:hover {{
        border-top: 4px solid {FG};
    }}

    /* ── Checkboxes & Radios ── */
    QCheckBox, QRadioButton {{
        spacing: 10px;
        padding: 4px 0;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1.5px solid {BORDER};
        border-radius: 4px;
        background-color: {BG_INPUT};
    }}
    QCheckBox::indicator:hover {{
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
        image: none;
        width: 10px;
        height: 10px;
        border: 3px solid {BG_INPUT};
    }}
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1.5px solid {BORDER};
        border-radius: 9px;
        background-color: {BG_INPUT};
    }}
    QRadioButton::indicator:hover {{
        border-color: {ACCENT};
    }}
    QRadioButton::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
        image: none;
        width: 8px;
        height: 8px;
        border: 4px solid {BG_INPUT};
    }}

    /* ── Lists ── */
    QListWidget {{
        background-color: {BG_INPUT};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        font-family: "{MONO}";
        font-size: 9pt;
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 8px 10px;
        border-radius: 6px;
        margin: 2px 0px;
    }}
    QListWidget::item:hover {{
        background-color: {BG_HOVER};
        color: {FG};
    }}
    QListWidget::item:selected {{
        background-color: {ACCENT};
        color: #ffffff;
        font-weight: 600;
    }}

    /* ── Scrollbars ── */
    QScrollBar:vertical {{
        background: {BG_PANEL};
        width: 10px;
        margin: 0px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ACCENT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        background: none;
        height: 0px;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    QScrollBar:horizontal {{
        background: {BG_PANEL};
        height: 10px;
        margin: 0px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER};
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {ACCENT};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        background: none;
        width: 0px;
    }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
    }}

    /* ── Progress Bar ── */
    QProgressBar {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        text-align: center;
        background: {BG_INPUT};
        color: {FG};
        font-weight: bold;
        font-size: 9pt;
        min-height: 20px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 6px;
    }}
    QLabel#statusLabel {{
        color: {FG_DIM};
        font-family: "{MONO}";
        font-size: 9pt;
    }}

    /* ── Custom UI Element Classes ── */
    QScrollArea#cropsScrollArea {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QScrollArea#cropsScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}
    QLabel#leftImageLabel {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QWidget#bottomStatusBar {{
        background-color: {BG_PANEL};
        border-top: 1px solid {BORDER};
    }}
    QLabel#pathLabel {{
        color: {FG_DIM};
        font-family: "{MONO}";
        font-size: 9pt;
    }}
    QFrame#editorCanvas {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* ── Result Tab Classify Buttons ── */
    QPushButton#resultPosButton {{
        background-color: #10b981;
        color: #0c0d12;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        min-height: 44px;
    }}
    QPushButton#resultPosButton:hover {{
        background-color: #34d399;
    }}
    QPushButton#resultPosButton:pressed {{
        background-color: #059669;
    }}

    QPushButton#resultNegButton {{
        background-color: #ef4444;
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        min-height: 44px;
    }}
    QPushButton#resultNegButton:hover {{
        background-color: #f87171;
    }}
    QPushButton#resultNegButton:pressed {{
        background-color: #dc2626;
    }}

    QPushButton#resultInvButton {{
        background-color: #f59e0b;
        color: #0c0d12;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        min-height: 44px;
    }}
    QPushButton#resultInvButton:hover {{
        background-color: #fbbf24;
    }}
    QPushButton#resultInvButton:pressed {{
        background-color: #d97706;
    }}
    """
