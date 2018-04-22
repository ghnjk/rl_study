#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random


class MazeEnv(object):
    def __init__(self, row_count, col_count, block_rate=0.3):
        self.row_count = row_count
        self.col_count = col_count
        self.init_pos = [random.randint(0, row_count - 1), random.randint(0, col_count - 1)]
        while True:
            self.target_pos = [random.randint(0, row_count - 1), random.randint(0, col_count - 1)]
            if self.target_pos == self.init_pos:
                continue
            else:
                break
        self.mp = []
        for r in range(self.row_count):
            self.mp.append([])
            for c in range(self.col_count):
                if random.uniform(0, 1) <= block_rate:
                    s = '#'
                else:
                    s = '.'
                if [r, c] == self.init_pos:
                    s = '.'
                elif [r, c] == self.target_pos:
                    s = 'T'
                self.mp[r].append(s)
        self.reset_env()

    def reset_env(self):
        self.cur_pos = list(self.init_pos)

    def __str__(self):
        res = ""
        for r in range(self.row_count):
            s = ""
            for c in range(self.col_count):
                if [r, c] == self.cur_pos:
                    s += "S"
                elif [r, c] == self.target_pos:
                    s += 'T'
                else:
                    s += self.mp[r][c]
            res += s + "\n"
        return res

    def get_map_state(self):
        s = []
        for r in range(self.row_count):
            for c in range(self.col_count):
                if [r, c] == self.cur_pos:
                    s.append(0)
                elif [r, c] == self.target_pos:
                    s.append(1)
                elif self.mp[r][c] == '#':
                    s.append(2)
                else:
                    s.append(3)
        return s

    def get_state(self):
        return (self.cur_pos[0] + 1) * (self.col_count + 2) + self.cur_pos[1] + 1

    def get_state_count(self):
        return (self.row_count + 2) * (self.col_count + 2)

    @staticmethod
    def get_all_actions():
        return ['R', 'L', 'U', 'D']

    def do_action(self, action):
        if action == 'R':
            self.cur_pos[1] += 1
        elif action == 'L':
            self.cur_pos[1] -= 1
        elif action == 'U':
            self.cur_pos[0] -= 1
        elif action == 'D':
            self.cur_pos[0] += 1
        else:
            print "invalid action: " + str(action)
        if self.cur_pos[0] < 0 or self.cur_pos[0] >= self.row_count:
            s = '#'
        elif self.cur_pos[1] < 0 or self.cur_pos[1] >= self.col_count:
            s = '#'
        else:
            s = self.mp[self.cur_pos[0]][self.cur_pos[1]]
        if s == '#':
            reward = -1
            is_terminate = True
        elif s == 'T':
            reward = 1
            is_terminate = True
        else:
            reward = 0
            is_terminate = False
        return self.get_state(), is_terminate, reward
