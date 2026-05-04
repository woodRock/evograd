"""
EvoGrad DSL Parser.

Parses a small domain-specific language into an IRGraph.

Grammar (EBNF):
  program     := block*
  block       := network_block | population_block | train_block
  network_block := 'network' NAME '{' network_body '}'
  network_body  := type_decl? layers_decl? forward_decl?
  type_decl     := 'type' ':' type_expr '->' type_expr
  layers_decl   := 'layers' '{' layer_stmt* '}'
  layer_stmt    := NAME '=' layer_expr ('|>' layer_expr)*
  layer_expr    := NAME '(' arg_list? ')'
  forward_decl  := 'forward' '=' NAME ('>>' NAME)*
  population_block := 'population' NAME '{' pop_body '}'
  pop_body      := genome_type_decl fitness_decl evolve_decl?
  train_block   := 'train' '{' train_body '}'

Example:
  network MLP {
    type: Float[batch, 784] -> Float[batch, 10]
    layers {
      fc1 = Dense(784, 256) |> ReLU
      fc2 = Dense(256, 128) |> ReLU
      out = Dense(128, 10)  |> Softmax
    }
    forward = fc1 >> fc2 >> out
  }

  population Searcher {
    genome_type: Float[128, 256]
    fitness: CrossEntropy(MLP)
    evolve {
      select = Tournament(k=7)
      crossover = Uniform(rate=0.5)
      mutate = Gaussian(sigma=0.01)
    }
  }

  train {
    mode = Hybrid(evolutionary: Searcher(50 generations),
                  gradient: Adam(lr=0.001, epochs=10))
  }
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ------------------------------------------------------------------ #
# Tokens
# ------------------------------------------------------------------ #

@dataclass
class Token:
    kind: str
    value: str
    line: int


_TOKEN_PATTERNS: List[Tuple[str, str]] = [
    ("COMMENT",  r"#[^\n]*"),
    ("NEWLINE",  r"\n"),
    ("WS",       r"[ \t]+"),
    ("ARROW",    r"->"),
    ("PIPE_FWD", r"\|>"),
    ("COMPOSE",  r">>"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("COLON",    r":"),
    ("COMMA",    r","),
    ("EQ",       r"="),
    ("INT",      r"\d+"),
    ("FLOAT",    r"\d+\.\d+"),
    ("NAME",     r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
]

_MASTER = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS)
)

_KEYWORDS = {
    "network", "population", "train", "type", "layers",
    "forward", "genome_type", "fitness", "evolve",
    "select", "crossover", "mutate", "mode", "generations",
    "epochs", "lr",
}


def tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    line = 1
    for m in _MASTER.finditer(src):
        kind = m.lastgroup
        value = m.group()
        if kind in ("WS", "COMMENT"):
            continue
        if kind == "NEWLINE":
            line += 1
            continue
        if kind == "NAME" and value in _KEYWORDS:
            kind = value.upper()
        tokens.append(Token(kind, value, line))
    return tokens


# ------------------------------------------------------------------ #
# AST nodes
# ------------------------------------------------------------------ #

@dataclass
class ASTNode:
    pass

@dataclass
class TypeExpr(ASTNode):
    dtype: str          # e.g. "Float"
    dims: List[str]     # e.g. ["batch", "784"]

@dataclass
class LayerExpr(ASTNode):
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]

@dataclass
class LayerStmt(ASTNode):
    var_name: str
    pipeline: List[LayerExpr]   # each |> stage

@dataclass
class NetworkBlock(ASTNode):
    name: str
    input_type: Optional[TypeExpr]
    output_type: Optional[TypeExpr]
    layers: List[LayerStmt]
    forward: List[str]          # ordered list of variable names

@dataclass
class EvolveSpec(ASTNode):
    select: Optional[LayerExpr]
    crossover: Optional[LayerExpr]
    mutate: Optional[LayerExpr]

@dataclass
class PopulationBlock(ASTNode):
    name: str
    genome_type: Optional[TypeExpr]
    fitness: Optional[LayerExpr]
    evolve: Optional[EvolveSpec]

@dataclass
class TrainBlock(ASTNode):
    mode: Optional[LayerExpr]

@dataclass
class Program(ASTNode):
    networks: List[NetworkBlock]
    populations: List[PopulationBlock]
    train: Optional[TrainBlock]


# ------------------------------------------------------------------ #
# Parser
# ------------------------------------------------------------------ #

class ParseError(SyntaxError):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    # ── Helpers ────────────────────────────────────────────────────

    def _peek(self) -> Optional[Token]:
        while self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _peek_kind(self) -> Optional[str]:
        t = self._peek()
        return t.kind if t else None

    def _consume(self, kind: str = None) -> Token:
        t = self._peek()
        if t is None:
            raise ParseError(f"Unexpected end of input, expected {kind!r}")
        if kind and t.kind != kind:
            raise ParseError(
                f"Line {t.line}: expected {kind!r}, got {t.kind!r} ({t.value!r})"
            )
        self._pos += 1
        return t

    def _maybe(self, kind: str) -> Optional[Token]:
        t = self._peek()
        if t and t.kind == kind:
            self._pos += 1
            return t
        return None

    # ── Top-level ──────────────────────────────────────────────────

    def parse(self) -> Program:
        networks: List[NetworkBlock] = []
        populations: List[PopulationBlock] = []
        train: Optional[TrainBlock] = None

        while self._peek() is not None:
            k = self._peek_kind()
            if k == "NETWORK":
                networks.append(self._parse_network())
            elif k == "POPULATION":
                populations.append(self._parse_population())
            elif k == "TRAIN":
                train = self._parse_train()
            else:
                t = self._consume()
                raise ParseError(
                    f"Line {t.line}: unexpected token {t.value!r}"
                )
        return Program(networks, populations, train)

    # ── Network block ───────────────────────────────────────────────

    def _parse_network(self) -> NetworkBlock:
        self._consume("NETWORK")
        name = self._consume("NAME").value
        self._consume("LBRACE")
        in_type = out_type = None
        layer_stmts: List[LayerStmt] = []
        forward: List[str] = []

        while self._peek_kind() != "RBRACE":
            k = self._peek_kind()
            if k == "TYPE":
                self._consume("TYPE")
                self._consume("COLON")
                in_type = self._parse_type_expr()
                self._consume("ARROW")
                out_type = self._parse_type_expr()
            elif k == "LAYERS":
                self._consume("LAYERS")
                self._consume("LBRACE")
                while self._peek_kind() != "RBRACE":
                    layer_stmts.append(self._parse_layer_stmt())
                self._consume("RBRACE")
            elif k == "FORWARD":
                self._consume("FORWARD")
                self._consume("EQ")
                forward = self._parse_compose_chain()
            else:
                t = self._consume()
                raise ParseError(f"Unexpected in network body: {t.value!r}")

        self._consume("RBRACE")
        return NetworkBlock(name, in_type, out_type, layer_stmts, forward)

    def _parse_type_expr(self) -> TypeExpr:
        dtype = self._consume("NAME").value
        self._consume("LBRACKET")
        dims: List[str] = []
        while self._peek_kind() != "RBRACKET":
            t = self._consume()
            if t.kind in ("NAME", "INT"):
                dims.append(t.value)
            # skip commas
        self._consume("RBRACKET")
        return TypeExpr(dtype, dims)

    def _parse_layer_stmt(self) -> LayerStmt:
        var = self._consume("NAME").value
        self._consume("EQ")
        stages: List[LayerExpr] = [self._parse_layer_expr()]
        while self._maybe("PIPE_FWD"):
            stages.append(self._parse_layer_expr())
        return LayerStmt(var, stages)

    def _parse_layer_expr(self) -> LayerExpr:
        name = self._consume("NAME").value
        if self._peek_kind() == "LPAREN":
            self._consume("LPAREN")
            args, kwargs = self._parse_args()
            self._consume("RPAREN")
        else:
            args, kwargs = [], {}
        return LayerExpr(name, args, kwargs)

    def _parse_args(self) -> Tuple[List[Any], Dict[str, Any]]:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        while self._peek_kind() != "RPAREN":
            # Check for kwarg
            if (self._peek_kind() == "NAME" and
                    self._pos + 1 < len(self._tokens) and
                    self._tokens[self._pos + 1].kind == "EQ"):
                key = self._consume("NAME").value
                self._consume("EQ")
                val = self._parse_value()
                kwargs[key] = val
            else:
                args.append(self._parse_value())
            self._maybe("COMMA")
        return args, kwargs

    def _parse_value(self) -> Any:
        t = self._peek()
        if t.kind == "FLOAT":
            self._consume()
            return float(t.value)
        elif t.kind == "INT":
            self._consume()
            return int(t.value)
        elif t.kind == "NAME":
            self._consume()
            return t.value
        else:
            self._consume()
            return t.value

    def _parse_compose_chain(self) -> List[str]:
        names = [self._consume("NAME").value]
        while self._maybe("COMPOSE"):
            names.append(self._consume("NAME").value)
        return names

    # ── Population block ────────────────────────────────────────────

    def _parse_population(self) -> PopulationBlock:
        self._consume("POPULATION")
        name = self._consume("NAME").value
        self._consume("LBRACE")
        genome_type = fitness = evolve = None

        while self._peek_kind() != "RBRACE":
            k = self._peek_kind()
            if k == "GENOME_TYPE":
                self._consume("GENOME_TYPE")
                self._consume("COLON")
                genome_type = self._parse_type_expr()
            elif k == "FITNESS":
                self._consume("FITNESS")
                self._consume("COLON")
                fitness = self._parse_layer_expr()
            elif k == "EVOLVE":
                evolve = self._parse_evolve()
            else:
                t = self._consume()
                raise ParseError(f"Unexpected in population body: {t.value!r}")

        self._consume("RBRACE")
        return PopulationBlock(name, genome_type, fitness, evolve)

    def _parse_evolve(self) -> EvolveSpec:
        self._consume("EVOLVE")
        self._consume("LBRACE")
        select = crossover = mutate = None
        while self._peek_kind() != "RBRACE":
            k = self._peek_kind()
            if k == "SELECT":
                self._consume("SELECT")
                self._consume("EQ")
                select = self._parse_layer_expr()
            elif k == "CROSSOVER":
                self._consume("CROSSOVER")
                self._consume("EQ")
                crossover = self._parse_layer_expr()
            elif k == "MUTATE":
                self._consume("MUTATE")
                self._consume("EQ")
                mutate = self._parse_layer_expr()
            else:
                t = self._consume()
                raise ParseError(f"Unexpected in evolve body: {t.value!r}")
        self._consume("RBRACE")
        return EvolveSpec(select, crossover, mutate)

    # ── Train block ─────────────────────────────────────────────────

    def _parse_train(self) -> TrainBlock:
        self._consume("TRAIN")
        self._consume("LBRACE")
        mode = None
        while self._peek_kind() != "RBRACE":
            k = self._peek_kind()
            if k == "MODE":
                self._consume("MODE")
                self._consume("EQ")
                mode = self._parse_layer_expr()
            else:
                t = self._consume()
                raise ParseError(f"Unexpected in train body: {t.value!r}")
        self._consume("RBRACE")
        return TrainBlock(mode)


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def parse(source: str) -> Program:
    """Parse an EvoGrad DSL string into an AST."""
    tokens = tokenize(source)
    return Parser(tokens).parse()
