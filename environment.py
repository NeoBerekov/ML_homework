import numpy
import gym


class Base:
    def __init__(self, x, y, ammo, fuel, hp, value, is_friendly):
        self.x = x
        self.y = y
        self.ammo = ammo
        self.fuel = fuel
        self.hp = hp
        self.value = value
        self.is_friendly = is_friendly


class Bomber:
    def __init__(self, ID, x, y, max_ammo, max_fuel):
        self.ID = ID
        self.x = x
        self.y = y
        self.max_ammo = max_ammo
        self.max_fuel = max_fuel


class CombatMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.dict_friendly_base = {}
        self.dict_enemy_base = {}
        self.dict_bomber = {}
        self.ammo_map = numpy.zeros((self.width, self.height))
        self.fuel_map = numpy.zeros((self.width, self.height))
        self.hp_map = numpy.zeros((self.width, self.height))
        self.value_map = numpy.zeros((self.width, self.height))
        self.is_passable_map = numpy.ones((self.width, self.height))

    def add_base(self, base):
        if base.is_friendly:
            self.dict_friendly_base[(base.x, base.y)] = base
        else:
            self.dict_enemy_base[(base.x, base.y)] = base

    def add_bomber(self, ID, x, y, max_ammo, max_fuel):
        self.dict_bomber[ID] = Bomber(ID, x, y, max_ammo, max_fuel)

    def add_friendly_base(self, x, y, ammo, fuel, hp, value):
        self.dict_friendly_base[(x, y)] = Base(x, y, ammo, fuel, hp, value, True)

    def add_enemy_base(self, x, y, ammo, fuel, hp, value):
        self.dict_enemy_base[(x, y)] = Base(x, y, ammo, fuel, hp, value, False)

    def create_ammo_map(self):
        for key in self.dict_friendly_base:
            self.ammo_map[key[0]][key[1]] = self.dict_friendly_base[key].ammo

    def create_fuel_map(self):
        for key in self.dict_friendly_base:
            self.fuel_map[key[0]][key[1]] = self.dict_friendly_base[key].fuel

    def create_hp_map(self):
        for key in self.dict_friendly_base:
            self.hp_map[key[0]][key[1]] = self.dict_enemy_base[key].hp

    def create_value_map(self):
        for key in self.dict_friendly_base:
            self.value_map[key[0]][key[1]] = self.dict_enemy_base[key].value

    def create_is_passable_map(self):
        for key in self.dict_enemy_base:
            self.is_passable_map[key[0]][key[1]] = 0



    def init_env(self, baseList):
        for base in baseList:
            self.add_base(base)
