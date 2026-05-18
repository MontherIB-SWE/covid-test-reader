"""Reusable Qt controls — consistent buttons and toolbar groups."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Single toolbar row height — must fit QSS padding + border + 10pt text on HiDPI.
BTN_HEIGHT = 40
BTN_HEIGHT_COMPACT = 36


def make_button(
    text: str,
    on_click: Callable[[], None] | None = None,
    *,
    style: str = "secondary",
    tooltip: str | None = None,
    min_width: int | None = None,
) -> QPushButton:
    """Create a styled push button (primary / secondary / danger / warning / compact)."""
    btn = QPushButton(text)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)

    name_map = {
        "primary": "accentButton",
        "secondary": "secondaryButton",
        "danger": "dangerButton",
        "warning": "warningButton",
        "compact": "compactButton",
    }
    btn.setObjectName(name_map.get(style, "secondaryButton"))

    height = BTN_HEIGHT_COMPACT if style == "compact" else BTN_HEIGHT
    btn.setFixedHeight(height)
    btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    if min_width is not None:
        btn.setMinimumWidth(min_width)
    if tooltip:
        btn.setToolTip(tooltip)
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn


def make_menu_button(
    text: str,
    actions: list[tuple[str, Callable[[], None] | None]],
    *,
    style: str = "secondary",
    tooltip: str | None = None,
) -> QPushButton:
    """Create a button with a dropdown menu. Use '-' for separators."""
    btn = make_button(text, style=style, tooltip=tooltip)
    menu = QMenu(btn)
    menu.setCursor(Qt.CursorShape.PointingHandCursor)
    for label, slot in actions:
        if label == "-":
            menu.addSeparator()
        elif slot:
            menu.addAction(label, slot)
    btn.setMenu(menu)
    return btn


def normalize_toolbar_widget(widget: QWidget, *, height: int = BTN_HEIGHT) -> QWidget:
    """Give combos, labels, etc. the same vertical size as toolbar buttons."""
    widget.setFixedHeight(height)
    widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    if isinstance(widget, QLabel):
        widget.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    elif isinstance(widget, QComboBox):
        widget.setMinimumContentsLength(0)
    return widget


def toolbar_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setObjectName("toolbarSeparator")
    line.setFixedWidth(1)
    line.setFixedHeight(BTN_HEIGHT)
    line.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return line


def tool_group(title: str, *widgets: QWidget) -> QWidget:
    """Labeled cluster of controls (e.g. 'Files' + Open buttons)."""
    wrap = QWidget()
    row = QHBoxLayout(wrap)
    row.setContentsMargins(0, 0, 8, 0)
    row.setSpacing(6)
    row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
    if title:
        lbl = QLabel(title)
        lbl.setObjectName("toolGroupLabel")
        row.addWidget(lbl)
    for w in widgets:
        row.addWidget(w)
    return wrap


def toolbar_panel(*rows: QWidget) -> QWidget:
    """Stack one or more horizontal toolbar rows on a panel background."""
    panel = QWidget()
    panel.setObjectName("toolbarPanel")
    col = QVBoxLayout(panel)
    col.setContentsMargins(12, 10, 12, 10)
    col.setSpacing(6)
    for row in rows:
        col.addWidget(row)
    return panel


def toolbar_row(*items: QWidget) -> QWidget:
    """Single horizontal toolbar row with stretch at the end."""
    row_w = QWidget()
    lay = QHBoxLayout(row_w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
    for item in items:
        lay.addWidget(item)
    lay.addStretch(1)
    return row_w
