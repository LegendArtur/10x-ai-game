from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests
import os
import threading

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

# Define weights for each unit type
weights = {
    'Virus': 3,
    'Tech': 3,
    'Firewall': 3,
    'Program': 3,
    'AI': 9999
}

class TimeLimitExceededException(Exception):
    pass

##############################################################################################################
# LOGGING

logfile = open('templog.txt', "w")

##############################################################################################################

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None
    heuristic_type: str | None = "e0"

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0
    non_root_nodes: int = 0
    non_leaf_nodes: int = 0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    h_player: Player = Player.Attacker
    next_player: Player = Player.Attacker
    turns_played : int = 1
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))
        #initialize the turns_played to start from 1
        self.turns_played = 0
    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        # Check if move is to an adjacent cell
        if coords.dst != coords.src and coords.dst not in coords.src.iter_adjacent():
            return False
        
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False
        
        dstunit = self.get(coords.dst)

        #Check if an AI, a Firewall or a Program 
        if unit.type.value == 0 or unit.type.value == 3 or unit.type.value == 4:
            # Check if the move is valid for the specific units
            if unit.player == Player.Attacker:
                # The attacker’s AI, Firewall and Program can only move up or left.
                if coords.dst.row == coords.src.row+1 or coords.dst.col == coords.src.col+1:
                    # If engaged in combat, should still be able to attack and repair.
                    if dstunit is not None and dstunit.player == Player.Defender:
                        return True
                    return False
            else:
                # The defender’s AI, Firewall and Program can only move down or right.
                if coords.dst.row == coords.src.row-1 or coords.dst.col == coords.src.col-1:
                    # If engaged in combat, should still be able to attack and repair.
                    if dstunit is not None and dstunit.player == Player.Attacker:
                        return True
                    return False
            
            # Check if wants to move but is engaged in combat
            if self.get(coords.dst) == None:
                for coord in coords.src.iter_adjacent():
                    if self.get(coord) is not None and self.get(coord).player != self.next_player:
                        return False
                    
        return True

    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if self.is_valid_move(coords):
            #self destruct
            unit = self.get(coords.src)
            target = self.get(coords.dst)
            if coords.src == coords.dst:
                area = coords.src.iter_range(1)
                for units in area:
                    if coords.src == units:
                        continue
                    temp = self.get(units)
                    if temp == None:
                        continue
                    temp.mod_health(-2)

                    self.remove_dead(units)
                unit.mod_health(-9)
                self.remove_dead(coords.src)
                return (True, coords.to_string())
            else:
                #standard movement or interaction
                if target == None:
                    self.set(coords.dst,self.get(coords.src))
                    self.set(coords.src,None)
                    return (True, coords.to_string())
                elif target.player == unit.player:
                    if unit.repair_amount(target) == 0 or target.health == 9:
                        return (False, "invalid move")
                    target.mod_health(unit.repair_amount(target))
                    return (True, coords.to_string())
                else:
                    #if combat, they damage each other
                    target.mod_health(-(unit.damage_amount(target)))
                    unit.mod_health(-(target.damage_amount(unit)))
                    #redundant if not target.is_alive():
                    self.remove_dead(coords.dst)
                    #redundant if not unit.is_alive():
                    self.remove_dead(coords.src)
                    return (True, coords.to_string())

        return (False,"invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def get_output(self) -> str:
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"

        return output

    def get_board(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.get_output() + self.get_board()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    logfile.write("\n")
                    logfile.write(f"Turn #{self.turns_played}: \n")
                    logfile.write(f"Player {self.next_player.name}: ")
                    print(f"Player {self.next_player.name}: ",end='')
                    logfile.write(result + "\n")
                    print(result + "\n")

                    logfile.write(self.get_board() + "\n\n")
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        self.h_player = self.next_player
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                logfile.write("\n")
                logfile.write(f"Turn #{self.turns_played}: \n")
                logfile.write(f"Computer {self.next_player.name}: ")
                print(f"Computer {self.next_player.name}: ",end='')
                logfile.write(result + "\n")
                print(result + "\n")

                logfile.write(self.get_board() + "\n\n")
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        elif self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        elif self._defender_has_ai:
            return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)
        

    def minimax_alpha_beta(self, game, depth, alpha, beta, maximizing_player):
        is_root_node = depth == self.options.max_depth  # Check if this is the root node
        is_leaf_node = depth == 0 or game.is_finished()  # Check if this is a leaf node

        # Count non-root nodes
        if not is_root_node:
            self.stats.non_root_nodes += 1

        # Count non-leaf nodes
        if not is_leaf_node:
            self.stats.non_leaf_nodes += 1

        # Existing code for leaf node evaluation
        if is_leaf_node:
            if self.options.heuristic_type == "e0":
                self.stats.evaluations_per_depth[self.options.max_depth - depth] = self.stats.evaluations_per_depth.get(self.options.max_depth - depth, 0) + 1
                return game.heuristic_e0()
            if self.options.heuristic_type == "e1":
                self.stats.evaluations_per_depth[self.options.max_depth - depth] = self.stats.evaluations_per_depth.get(self.options.max_depth - depth, 0) + 1
                return game.heuristic_e1()
            if self.options.heuristic_type == "e2":
                self.stats.evaluations_per_depth[self.options.max_depth - depth] = self.stats.evaluations_per_depth.get(self.options.max_depth - depth, 0) + 1
                return game.heuristic_e2()

        if maximizing_player:
            v = float('-inf')
            
            for move in game.move_candidates():
                child_game = game.clone()
                (success, result) = child_game.perform_move(move)
                if not success:
                    continue
                child_game.next_turn()
                eval_value = self.minimax_alpha_beta(child_game, depth - 1, alpha, beta, False)
                v = max(v, eval_value)
                alpha = max(alpha, v)
                if beta <= alpha:
                    break  # Beta cut-off
            return v
        else:
            v = float('inf')
            
            for move in game.move_candidates():
                child_game = game.clone()
                (success, result) = child_game.perform_move(move)
                if not success:
                    continue
                child_game.next_turn()
                eval_value = self.minimax_alpha_beta(child_game, depth - 1, alpha, beta, True)
                v = min(v, eval_value)
                beta = min(beta, v)
                if beta <= alpha:
                    break  # Alpha cut-off
            return v

    def get_best_move(self, depth):
        best_move = None
        max_eval = float('-inf')

        stop_search = threading.Event()  # Event to signal the thread to stop

        def worker():
            nonlocal best_move, max_eval

            for move in self.move_candidates():
                # Check if we should stop searching due to time limit
                if stop_search.is_set():
                    # Can remove this output in the future, but just to show it does respect time limit
                    print(f"Time limit of {self.options.max_time} seconds reached! Returning current values.")
                    break

                child_game = self.clone()
                (success, result) = child_game.perform_move(move)
                if not success:
                    continue
                child_game.next_turn()
                v = self.minimax_alpha_beta(child_game, depth - 1, float('-inf'), float('inf'), False)

                if v > max_eval:
                    max_eval = v
                    best_move = move


        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=(self.options.max_time - 1)) # - 1 second for time for the rest of the turn logic to be performed

        if thread.is_alive():
            stop_search.set()  # Signal the thread to stop searching
            thread.join()  # Wait for the thread to actually finish

        return max_eval, best_move
    
    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        try:
            (score, move) = self.get_best_move(self.options.max_depth)
        except TimeLimitExceededException:
            pass
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        logfile.write(f"Heuristic score: {score}\n")
        total_evals = sum(self.stats.evaluations_per_depth.values())
        print(f"Cumulative evaluations: {total_evals}")
        logfile.write(f"Cumulative evaluations: {total_evals}\n")
        #Average branching factor 
        total_depths = len(self.stats.evaluations_per_depth)
        if self.stats.non_leaf_nodes > 0:
            average_branching_factor = self.stats.non_root_nodes/self.stats.non_leaf_nodes if total_depths else 0
        else: 
            average_branching_factor = 0
        print(f"Average branching factor: {average_branching_factor:0.1f}")
        logfile.write(f"Average branching factor: {average_branching_factor:0.1f}\n")
        self.stats.non_leaf_nodes = 0
        self.stats.non_root_nodes = 0
        print(f"Evals per depth: ",end='')
        logfile.write(f"Evals per depth: ")
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
            logfile.write(f"{k}:{self.stats.evaluations_per_depth[k]} ")
        print()
        logfile.write("\n")
        print(f"Cumulative evals per depth: ",end='')
        logfile.write(f"Cumulative evals per depth: ")
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{(self.stats.evaluations_per_depth[k]/total_evals)*100:0.1f}% ",end='')
            logfile.write(f"{k}:{(self.stats.evaluations_per_depth[k]/total_evals)*100:0.1f}% ")
        print()
        logfile.write("\n")
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
            logfile.write(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        logfile.write("\n")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        logfile.write(f"Elapsed time: {elapsed_seconds:0.1f}s\n")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None
    
    def heuristic_e2(self) -> int:
        player1_score = 0
        player2_score = 0
        
        WIN_SCORE = 999999  # A very high score for winning
        LOSS_SCORE = -999999  # A very high negative score for losing
        MOVE_TOWARDS_AI_WEIGHT = 6000
        HEALTH_FACTOR = 100  # Adjust as needed to increase/decrease the influence of health
        AI_HEALTH_WEIGHT = 2000 # High factor for AI health so that units are motivated to go for AI.
        
        ai_location_opponent = None
        ai_location_self = None
        
        # Check opponent's AI
        for (coord, unit) in self.player_units(Player.Defender if self.h_player == Player.Attacker else Player.Attacker):
            if unit.type.name == "AI":
                ai_location_opponent = coord
                player2_score += unit.health * AI_HEALTH_WEIGHT  
                break
        
        # Check own AI
        for (coord, unit) in self.player_units(self.h_player):
            if unit.type.name == "AI":
                ai_location_self = coord
                player1_score += unit.health * AI_HEALTH_WEIGHT
                break

        # If the opponent's AI is not found, it means the current player (AI) has won
        if not ai_location_opponent:
            return WIN_SCORE
        
        # If the AI's own unit is not on the board, it means the AI has lost
        if not ai_location_self:
            return LOSS_SCORE

        # For the current player
        for (coord, unit) in self.player_units(self.h_player):
            # Incentive to move closer to the opponent's AI
            distance_to_ai = abs(coord.row - ai_location_opponent.row) + abs(coord.col - ai_location_opponent.col)
            if distance_to_ai == 0:
                player1_score += MOVE_TOWARDS_AI_WEIGHT
            else:
                player1_score += MOVE_TOWARDS_AI_WEIGHT / distance_to_ai

            # Health consideration
            if unit.type.name != "AI":
                player1_score += unit.health * HEALTH_FACTOR

        # For the opposing player (similar considerations but for the opponent)
        for (coord, unit) in self.player_units(Player.Defender if self.h_player == Player.Attacker else Player.Attacker):
            distance_to_ai = abs(coord.row - ai_location_self.row) + abs(coord.col - ai_location_self.col)
            if distance_to_ai == 0:
                player2_score += MOVE_TOWARDS_AI_WEIGHT  # Give maximum score if the unit is already on the AI's position
            else:
                player2_score += MOVE_TOWARDS_AI_WEIGHT / distance_to_ai

            if unit.type.name != "AI":
                player2_score += unit.health * HEALTH_FACTOR

        return int(player1_score - player2_score)


    def heuristic_e1(self) -> int:
        # Initialize the scores for both players
        player1_score = player2_score = 0

        # Define weights for each unit type
        weights = {
            'Virus': 3,
            'Tech': 3,
            'Firewall': 3,
            'Program': 3,
            'AI': 9999
        }

        for (coord, unit) in self.player_units(self.h_player):
            if unit is not None:
                unit_type = unit.type.name
                # Add the unit's health to its corresponding unit type score in HealthScore
                # Add the unit's health to its player's score
                player1_score += weights[unit_type] * unit.health

        for (coord, unit) in self.player_units(Player.Defender if self.h_player == Player.Attacker else Player.Attacker):
            if unit is not None:
                unit_type = unit.type.name
                # Add the unit's health to its corresponding unit type score in HealthScore
                # Add the unit's health to its player's score
                player2_score += weights[unit_type] * unit.health

        # Calculate the final score
        return player1_score - player2_score

    
    def heuristic_e0(self) -> int:
        # Initialize the scores for both players
        player1_score = player2_score = 0

        # Determine who 

        unitCount = {
            'Virus': 0,
            'Tech': 0,
            'Firewall': 0,
            'Program': 0,
            'AI': 0
        }

        # Get the number of each unit type for Player 1 (performing move)
        for (coord, unit) in self.player_units(self.h_player):
            # Add the unit's count to its corresponding unit type score in unitCount
            unitCount[unit.type.name] += 1
        
        for unitType in unitCount:
            player1_score += weights[unitType] * unitCount[unitType]
        
        # Reset unitCount
        unitCount = {
            'Virus': 0,
            'Tech': 0,
            'Firewall': 0,
            'Program': 0,
            'AI': 0
        }

        # Get the number of each unit type for Player 2 (not performing move)
        for (coord, unit) in self.player_units(Player.Defender if self.h_player == Player.Attacker else Player.Attacker):
            # Add the unit's count to its corresponding unit type score in unitCount
            unitCount[unit.type.name] += 1
        
        for unitType in unitCount:
            player2_score += weights[unitType] * unitCount[unitType]

        return player1_score - player2_score

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--max_moves', type=float, help='maximum moves per game')
    parser.add_argument('--game_type', type=str, default="attacker", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--alpha_beta', type=bool, help='if a player is an AI, whether alpha-beta is on or off')
    parser.add_argument('--heuristic_type', type=str, help='heuristic type: e0|e1|e2')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    logfileName = f"gameTrace-{args.alpha_beta}-{args.max_time}-{int(args.max_moves)}.txt"
    counter = 0
    while os.path.exists(logfileName):
        logfileName = f"gameTrace-{args.alpha_beta}-{args.max_time}-{int(args.max_moves)}-{counter}.txt"
        counter += 1

    logfile.write(f"1. The game parameters:\n\ta. Timeout (in s): {args.max_time}\n\tb. Max number of turns: {args.max_moves}\n",)

    
    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
        logfile.write(f"\tc. Play mode: Attacker (Human) vs Defender (AI)\n")
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
        logfile.write(f"\tc. Play mode: Attacker (AI) vs Defender (Human)\n")
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
        logfile.write(f"\tc. Play mode: Attacker (Human) vs Defender (Human)\n")
    else:
        game_type = GameType.CompVsComp
        logfile.write(f"\tc. Play mode: Attacker (AI) vs Defender (AI)\n")

    if not args.game_type == "manual":
        logfile.write(f"\td. Alpha-Beta is:")
        logfile.write(f" {'ON' if args.alpha_beta else 'OFF'}\n")
        logfile.write(f"\te. Heuristic: {args.heuristic_type}\n",)

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.heuristic_type is not None:
        options.heuristic_type = args.heuristic_type
    if args.max_moves is not None:
        options.max_turns = int(args.max_moves)
    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta

    # create a new game
    game = Game(options=options)

    logfile.write(f"\n2. Initialtial game board:\n{game.get_board()}\n3. Gameplay trace:\n\n")

    # the main game loop
    try:
        while True:
            print()
            print(game)
            winner = game.has_winner()
            if game.turns_played == game.options.max_turns:
                print(f"Tie, max number of turns played! {game.turns_played}/{game.options.max_turns}")
                logfile.write(f"Tie, max number of turns played! {game.turns_played}/{game.options.max_turns}")
                logfile.close()
                os.rename('templog.txt', logfileName)
                break
            if winner is not None:
                print(f"{winner.name} wins in {game.turns_played} turns!")
                logfile.write(f"{winner.name} wins in {game.turns_played} turns!\n")
                logfile.close()
                os.rename('templog.txt', logfileName)
                break
            if game.options.game_type == GameType.AttackerVsDefender:
                game.human_turn()
            elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
                game.human_turn()
            elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
                game.human_turn()
            else:
                player = game.next_player
                move = game.computer_turn()
                if move is not None:
                    game.post_move_to_broker(move)
                else:
                    print("Computer doesn't know what to do!!!")
                    logfile.close()
                    os.rename('templog.txt', logfileName)
                    exit(1)
    except KeyboardInterrupt:
        print("Game interrupted by user.")
        logfile.write("Game interrupted by user.\n")
        logfile.close()
        os.rename('templog.txt', logfileName)
##############################################################################################################

if __name__ == '__main__':
    main()

