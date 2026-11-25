"""Outils de dessin : polices disponibles, style visuel tiré au hasard, canevas A4 ou ticket."""

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# A4 à 150 DPI : assez fin pour l'OCR de la partie 2, assez léger pour l'entraînement.
A4 = (1240, 1754)
TICKET_WIDTH = 576  # ticket thermique 80 mm

FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
]
SANS = [
    ("arial.ttf", "arialbd.ttf"),
    ("calibri.ttf", "calibrib.ttf"),
    ("verdana.ttf", "verdanab.ttf"),
    ("tahoma.ttf", "tahomabd.ttf"),
    ("segoeui.ttf", "segoeuib.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
]
SERIF = [
    ("times.ttf", "timesbd.ttf"),
    ("georgia.ttf", "georgiab.ttf"),
    ("cambria.ttc", "cambriab.ttf"),
    ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
]
MONO = [
    ("cour.ttf", "courbd.ttf"),
    ("consola.ttf", "consolab.ttf"),
    ("lucon.ttf", "lucon.ttf"),
    ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"),
    ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
]


@lru_cache
def _find(filename: str) -> Path | None:
    for directory in FONT_DIRS:
        if not directory.is_dir():
            continue
        direct = directory / filename
        if direct.is_file():
            return direct
        for match in directory.rglob(filename):
            return match
    return None


@lru_cache
def available(families: tuple[tuple[str, str], ...]) -> list[tuple[Path, Path]]:
    found = []
    for regular, bold in families:
        reg_path, bold_path = _find(regular), _find(bold)
        if reg_path:
            found.append((reg_path, bold_path or reg_path))
    return found


@lru_cache(maxsize=512)
def load_font(path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    if path is None:  # aucune police système : police intégrée à Pillow
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(str(path), size)


@dataclass
class Style:
    regular: Path | None
    bold: Path | None
    accent: tuple[int, int, int]
    ink: tuple[int, int, int]
    table: str  # "grille", "bandes", "lignes"
    header_fill: bool
    size: int  # taille de base du texte

    @classmethod
    def random(cls, rng: random.Random, families: str = "doc") -> "Style":
        pool = {
            "doc": available(tuple(SANS)) + available(tuple(SERIF)),
            "mono": available(tuple(MONO)),
        }[families]
        regular, bold = rng.choice(pool) if pool else (None, None)
        accent = rng.choice(
            [
                (20, 60, 140),
                (150, 20, 30),
                (20, 110, 70),
                (60, 60, 60),
                (200, 90, 0),
                (90, 40, 140),
                (0, 120, 160),
                (0, 0, 0),
            ]
        )
        return cls(
            regular=regular,
            bold=bold,
            accent=accent,
            ink=rng.choice([(0, 0, 0), (25, 25, 25), (40, 40, 50)]),
            table=rng.choice(["grille", "bandes", "lignes"]),
            header_fill=rng.random() < 0.6,
            size=rng.randint(19, 24),
        )


class Canvas:
    def __init__(self, style: Style, size: tuple[int, int] = A4):
        self.style = style
        self.img = Image.new("RGB", size, "white")
        self.draw = ImageDraw.Draw(self.img)
        self.width, self.height = size

    def font(self, size: int | None = None, bold: bool = False) -> ImageFont.FreeTypeFont:
        path = self.style.bold if bold else self.style.regular
        return load_font(path, size or self.style.size)

    def text(
        self,
        xy,
        text: str,
        size: int | None = None,
        bold: bool = False,
        color=None,
        anchor: str = "la",
    ) -> None:
        self.draw.text(
            xy, text, font=self.font(size, bold), fill=color or self.style.ink, anchor=anchor
        )

    def lines(
        self,
        x: int,
        y: int,
        lines: list[str],
        size: int | None = None,
        bold: bool = False,
        color=None,
        anchor: str = "la",
        spacing: float = 1.35,
    ) -> int:
        """Écrit plusieurs lignes et renvoie l'ordonnée sous le bloc."""
        size = size or self.style.size
        for line in lines:
            self.text((x, y), line, size, bold, color, anchor)
            y += int(size * spacing)
        return y

    def width_of(self, text: str, size: int | None = None, bold: bool = False) -> float:
        return self.draw.textlength(text, font=self.font(size, bold))

    def wrap(self, text: str, max_width: int, size: int | None = None) -> list[str]:
        words, lines, current = text.split(), [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if self.width_of(candidate, size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def paragraph(
        self,
        x: int,
        y: int,
        text: str,
        max_width: int,
        size: int | None = None,
        spacing: float = 1.4,
    ) -> int:
        return self.lines(x, y, self.wrap(text, max_width, size), size, spacing=spacing)

    def rule(self, x1: int, y: int, x2: int, width: int = 2, color=None) -> None:
        self.draw.line([(x1, y), (x2, y)], fill=color or self.style.ink, width=width)

    def box(self, xy, outline=None, fill=None, width: int = 2) -> None:
        self.draw.rectangle(xy, outline=outline or self.style.ink, fill=fill, width=width)

    def logo(self, x: int, y: int, initials: str, rng: random.Random, size: int = 90) -> None:
        """Petit logo géométrique (aucune marque réelle)."""
        color = self.style.accent
        shape = rng.choice(["rond", "carre", "losange", "barre"])
        if shape == "rond":
            self.draw.ellipse([x, y, x + size, y + size], fill=color)
        elif shape == "carre":
            self.draw.rounded_rectangle([x, y, x + size, y + size], radius=size // 6, fill=color)
        elif shape == "losange":
            h = size // 2
            self.draw.polygon(
                [(x + h, y), (x + size, y + h), (x + h, y + size), (x, y + h)], fill=color
            )
        else:
            self.draw.rectangle([x, y + size // 3, x + size, y + 2 * size // 3], fill=color)
            initials = ""
        if initials:
            self.text(
                (x + size / 2, y + size / 2),
                initials[:2].upper(),
                int(size * 0.4),
                True,
                (255, 255, 255),
                anchor="mm",
            )

    def table(
        self,
        x: int,
        y: int,
        columns: list[tuple[str, int, str]],
        rows: list[list[str]],
        size: int | None = None,
        row_height: int | None = None,
    ) -> int:
        """Tableau simple. `columns` = (titre, largeur, alignement l/r/c). Renvoie y en bas."""
        size = size or self.style.size - 2
        row_height = row_height or int(size * 2)
        total = sum(width for _, width, _ in columns)
        style, accent = self.style.table, self.style.accent

        def cells(values, yy, bold=False, color=None):
            cx = x
            for value, (_, width, align) in zip(values, columns, strict=True):
                pad = 10
                if align == "r":
                    self.text(
                        (cx + width - pad, yy + row_height / 2), value, size, bold, color, "rm"
                    )
                elif align == "c":
                    self.text((cx + width / 2, yy + row_height / 2), value, size, bold, color, "mm")
                else:
                    self.text((cx + pad, yy + row_height / 2), value, size, bold, color, "lm")
                cx += width

        header_color = None
        if self.style.header_fill:
            self.box([x, y, x + total, y + row_height], outline=accent, fill=accent)
            header_color = (255, 255, 255)
        cells([title for title, _, _ in columns], y, bold=True, color=header_color)
        if not self.style.header_fill:
            self.rule(x, y + row_height, x + total, 3, accent)
        y += row_height
        for i, row in enumerate(rows):
            if style == "bandes" and i % 2 == 0:
                self.box(
                    [x, y, x + total, y + row_height], outline=(238, 238, 238), fill=(238, 238, 238)
                )
            cells(row, y)
            if style == "grille":
                self.box([x, y, x + total, y + row_height], width=1)
                cx = x
                for _, width, _ in columns[:-1]:
                    cx += width
                    self.draw.line([(cx, y), (cx, y + row_height)], fill=self.style.ink, width=1)
            elif style == "lignes":
                self.rule(x, y + row_height, x + total, 1, (180, 180, 180))
            y += row_height
        return y
