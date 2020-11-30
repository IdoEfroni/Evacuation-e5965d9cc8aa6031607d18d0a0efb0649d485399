from Agent import *
from Point import Point

import pygame
from pygame.locals import *
from pygame.color import *
import thinkplot
import random
import time

from datetime import datetime
import pickle
import pandas

import sys

DEBUG = False


class Wall:
    def __init__(self, wallType,isReal=True,isRight=False, **parameters):
        # type : "line", points
        # Line : type='line'{ "pf: Point(x1,y1), "p2": Point(x2,y2) }
        self.wallType = wallType
        self.parameters = parameters
        self.isReal = isReal
        self.isRight = isRight
        self.checkValid()


    def checkValid(self):
        if self.wallType == 'line':
            assert isinstance(self.parameters['p1'], Point)
            assert isinstance(self.parameters['p2'], Point)


class Goal(Wall):
    """ Defines a goal. Currently, only horizontal and vertical lines are supported. """

    def checkValid(self):
        assert self.wallType == 'line'
        assert isinstance(self.parameters['p1'], Point)
        assert isinstance(self.parameters['p2'], Point)
        # assert (self.parameters['p1'].x == self.parameters['p2'].x or self.parameters['p1'].y == self.parameters[
        #     'p2'].y)

        # p1 should always be smaller than p2
        if (self.parameters['p1'].x == self.parameters['p2'].x):
            if self.parameters['p1'].y > self.parameters['p2'].y:
                p1Temp = self.parameters['p1']
                self.parameters['p1'] = self.parameters['p2']
                self.parameters['p2'] = p1Temp
        elif (self.parameters['p1'].y == self.parameters['p2'].y):
            if self.parameters['p1'].x > self.parameters['p2'].x:
                p1Temp = self.parameters['p1']
                self.parameters['p1'] = self.parameters['p2']
                self.parameters['p2'] = p1Temp

    def get_x(self):
        return (self.parameters['p1'].x)
    def get_higher_y(self):
        return (self.parameters['p2'].y)
    def get_lower_y(self):
        return (self.parameters['p1'].y)
    def is_right(self):
        if self.get_x()!=0:
            return True
        else:
            return False

    def agentsDistance(agent, other):
        return agent.pos._distance_(other.pos);



def agentsDistance(agent, other):
    return agent.pos.__distance__(other.pos);


def closeNeighbour(env, agent):
    for goal in env.goals:
        if agent.pos.__distance__(goal.parameters['p1']) <= 5 or agent.pos.__distance__(goal.parameters['p2']) <= 5:
            agent.certainty = True
            return goal
    xSum = 0
    ySum = 0
    totalNeigh = 0
    for other in env.agents:
        if agent.index == other.index:
            continue
        if agentsDistance(agent, other) <= 5:
            agent.certainty = True
            xSum += other.pos.x
            ySum += other.pos.y
            totalNeigh +=1

    if totalNeigh>0:
        return Goal('line',False, **{'p1': Point(xSum/totalNeigh,ySum/totalNeigh),
                                   'p2': Point(xSum/totalNeigh,ySum/totalNeigh)})
    else:
        agent.certainty = False
        return Goal('line',False, **{'p1': Point(randFloat(.5, 2 * 15 / 3 - .5), randFloat(.5, 15 - .5)),
                           'p2': Point(randFloat(.5, 2 * 15 / 3 - .5), randFloat(.5, 15 - .5))})  # random direction


def isFinished(agent, goals):

    if agent.pos.x >= agent.goal.parameters['p1'].x and agent.goal.isReal and agent.goal.isRight:
        return True
    elif agent.pos.x <= agent.goal.parameters['p1'].x and agent.goal.isReal and not agent.goal.isRight:
        return True
    return False


class Environment():
    conditions = {'k': 1.2 * 10 ** 5, 'ka': 2.4 * 10 ** 5}

    def __init__(self, N, walls, goals, agents, conditions, instruments, smoke):
        self.N = N
        self.walls = walls
        self.goals = goals
        self.agents = agents
        self.instruments = instruments
        # Conditions: Agent force, Agent repulsive distance, acceleration time, step length,
        self.conditions.update(conditions)
        self.smoke = smoke

    def step(self):
        for agent in self.agents:
            # print(agent.desiredDirection)
            selfDriveForce = agent.selfDriveForce()
            pairForce = Point(0, 0)
            wallForce = Point(0, 0)
            for wall in self.walls:
                wallForce += agent.wallForce(wall)
            for agent2 in self.agents:
                if agent.index == agent2.index:
                    continue
                pairForce += agent.pairForce(agent2)
            netForce = selfDriveForce + pairForce + wallForce
            agent.move(netForce, closeNeighbour(self, agent)) if self.smoke else agent.move(netForce)
            if isFinished(agent,self.goals):
                agent.Finish = True

        self.updateInstruments()

    def updateInstruments(self):
        for instrument in self.instruments:
            instrument.update(self)

    def plot(self, num):
        self.instruments[num].plot()


class EnvironmentViewer():
    BG_COLOR = Color(0, 0, 0)

    BLACK = Color(0, 0, 0)
    WHITE = Color(255, 255, 255)
    YELLOW = Color(255, 233, 0)
    RED = Color(203, 20, 16)
    GOAL = Color(252, 148, 37)

    pygameScale = 50

    def __init__(self, environment):
        self.env = environment
        self.screen = pygame.display.set_mode((800, 800))

    def draw(self):
        pygame.event.get()
        self.screen.fill(self.BG_COLOR)

        for agent in self.env.agents:
            self.drawAgent(agent)

        for wall in self.env.walls:
            self.drawWall(wall)

        for goal in self.env.goals:
            self.drawGoal(goal)

        pygame.display.update()

    def drawAgent(self, agent):
        # Draw agent
        pygame.draw.circle(self.screen, self.YELLOW, agent.pos.pygame, int(agent.size * self.pygameScale))
        # Draw desired vector
        pygame.draw.line(self.screen, self.YELLOW, agent.pos.pygame, (agent.pos + (agent.desiredDirection)).pygame)
        # Draw number in agent
        pygame.font.init()
        font = pygame.font.SysFont('arial', 20)
        text = font.render(str(agent.index), True, (255, 0, 0))
        self.screen.blit(text, Point(agent.pos.x-agent.size/2,agent.pos.y-agent.size).pygame)

        if (DEBUG): print("drew agent at ", agent.pos)

    def drawWall(self, wall, color=WHITE):

        if wall.wallType == 'line':
            pygame.draw.line(self.screen, color, wall.parameters['p1'].pygame, wall.parameters['p2'].pygame, 10)
            if (DEBUG): print("drew wall between {} and {}".format(wall.parameters['p1'], wall.parameters['p2']))

    def drawGoal(self, goal):
        self.drawWall(goal, color=self.GOAL)


class Instrument():
    """ Instrument that logs the state of the environment"""

    def __init__(self):
        self.metric = []

    def plot(self, **options):
        thinkplot.plot(self.metric, **options)
        thinkplot.show()


class ReachedGoal(Instrument):
    """ Logs the number of agents that have escaped """

    def update(self, env):
        self.metric.append(self.countReachedGoal(env))

    def countReachedGoal(self, env):
        num_escaped = 0

        for agent in env.agents:
            if agent.Finish:
                num_escaped += 1

        return num_escaped


def randFloat(minVal, maxVal):
    return random.random() * (maxVal - minVal) + minVal


def runSimulation(roomHeight=10,
                  roomWidth=8,
                  doorWidth=1,
                  numAgents=10,
                  agentMass=80,
                  desiredSpeed=1.5,
                  view=False,
                  smoke=False,
                  twoDoors = False,
                  halfMode = False):
    walls = []
    walls.append(Wall('line', **{'p1': Point(0, 0), 'p2': Point(roomWidth, 0)}))  # Top
    if not twoDoors:
        walls.append(Wall('line', **{ 'p1': Point(0,0), 'p2': Point(0, roomHeight) })) # Left
    walls.append(Wall('line', **{'p1': Point(0, roomHeight), 'p2': Point(roomWidth, roomHeight)}))  # Bottom

    walls.append(Wall('line', **{'p1': Point(roomWidth, 0),
                                 'p2': Point(roomWidth, roomHeight / 2 - doorWidth / 2)}))  # Top Doorway
    walls.append(Wall('line', **{'p1': Point(roomWidth, roomHeight / 2 + doorWidth / 2),
                                 'p2': Point(roomWidth, roomHeight)}))  # Bottom Doorway
    #door right
    # walls.append(Wall('line', **{ 'p1': Point(roomWidth,0), 'p2': Point(roomWidth, roomHeight/2 - doorWidth/2) })) # Top Doorway
    # walls.append(Wall('line', **{ 'p1': Point(roomWidth, roomHeight/2 + doorWidth/2), 'p2': Point(roomWidth, roomHeight) })) # Bottom Doorway

    #door left
    if twoDoors:
        walls.append(Wall('line', **{ 'p1': Point(0,0), 'p2': Point(0, roomHeight/2 - doorWidth/2) })) # Top Doorway
        walls.append(Wall('line', **{ 'p1': Point(0, roomHeight/2 + doorWidth/2), 'p2': Point(0, roomHeight) })) # Bottom Doorway

    goals = []
    # goals.append(Goal('line',True,True, **{'p1': Point(roomWidth, roomHeight / 2 - doorWidth / 2),
    # 'p2': Point(roomWidth, roomHeight / 2 + doorWidth / 2)}))
    goals.append(Goal('line',True,True, **{ 'p1': Point(roomWidth, roomHeight/2 - doorWidth/2), 'p2': Point(roomWidth, roomHeight/2 + doorWidth/2) }))
    if twoDoors:
        goals.append(Goal('line',True,False, **{ 'p1': Point(0, roomHeight/2 - doorWidth/2), 'p2': Point(0, roomHeight/2 + doorWidth/2) }))


    instruments = []
    instruments.append(ReachedGoal())

    agents = []
    # old_agent = int(numAgents*0.2)
    # for _ in range(old_agent):
    #     # Agent(size, mass, pos, goal, desiredSpeed = 4))
    #     size = randFloat(.25, .35)
    #     mass = agentMass
    #     pos = Point(randFloat(.5, 2*roomWidth/3 - .5), randFloat(.5,roomHeight-.5))
    #     goal = goals[0]
    #
    #     agents.append(Agent(size, mass, pos, goal, desiredSpeed=desiredSpeed/3))

    def closest_door(x,y,goal1,goal2):
        if y>goal1.get_higher_y():
            distance_left = math.sqrt(pow(x-goal2.get_x(),2) +pow(y-goal2.get_higher_y(),2) )
            distance_right = math.sqrt(pow(x-goal1.get_x(),2) +pow(y-goal1.get_higher_y(),2) )
        else:
            distance_left = math.sqrt(pow(x - goal2.get_x(), 2) + pow(y - goal2.get_lower_y(), 2))
            distance_right = math.sqrt(pow(x - goal1.get_x(), 2) + pow(y - goal1.get_lower_y(), 2))

        if distance_left < distance_right:
            return goal2
        else:
            return goal1
    #3.3
    i=1
    for _ in range(numAgents):
        # Agent(size, mass, pos, goal, desiredSpeed = 4))
        size = randFloat(.25, .35)
        mass = agentMass
        # pos = Point(randFloat(.5, 2 * roomWidth / 3 - .5), randFloat(.5, roomHeight - .5))
        pos = Point(randFloat(.5, roomWidth - .5), randFloat(.5, roomHeight - .5))
        if twoDoors:
            if halfMode:
                if (i < numAgents / 2):
                    goal = goals[0]
                else:
                    goal = closest_door(pos.x, pos.y, goals[0], goals[1])
            elif not halfMode and smoke:
                goal = closest_door(pos.x, pos.y, goals[0], goals[1]) if i<numAgents/2 else Goal('line', False, **{
                    'p1': Point(randFloat(.5, 2 * roomWidth / 3 - .5), randFloat(.5, roomHeight - .5)),
                    'p2': Point(randFloat(.5, 2 * roomWidth / 3 - .5), randFloat(.5, roomHeight - .5))})
            else:
                goal = closest_door(pos.x,pos.y,goals[0],goals[1])
        else:
            goal = goals[0] if not smoke else Goal('line', False, **{
                'p1': Point(randFloat(.5, 2 * roomWidth / 3 - .5), randFloat(.5, roomHeight - .5)),
                'p2': Point(randFloat(.5, 2 * roomWidth / 3 - .5), randFloat(.5, roomHeight - .5))})

        agents.append(Agent(size, mass, pos, goal, desiredSpeed=desiredSpeed))
        i+=1

    env = Environment(100, walls, goals, agents, {}, instruments, smoke)

    #3.1
    # i=1
    # for _ in range(numAgents):
    #     # Agent(size, mass, pos, goal, desiredSpeed = 4))
    #     size = randFloat(.25, .35)
    #     mass = agentMass
    #     pos = Point(randFloat(.5, 2*roomWidth/3 - .5), randFloat(.5,roomHeight-.5))
    #     print(pos)
    #     goal = closest_door(pos.x,pos.y,goals[0],goals[1])
    #     print(goal.get_x())
    #     #choice = random.randint(0,len(goals)-1)
    #     #print(choice)
    #     #goal = goals[choice]
    #     agents.append(Agent(size, mass, pos, goal, desiredSpeed=desiredSpeed))


    #3.2
    # size = randFloat(.25, .35)
    # mass = agentMass
    # pos = Point(randFloat(.5, 2 * roomWidth / 3 - .5), randFloat(.5, roomHeight - .5))
    # print(pos)
    # if (i < numAgents / 2):
    #     goal = goals[0]
    # else:
    #     goal = closest_door(pos.x, pos.y, goals[0], goals[1])
    #
    # print(goal.get_x())
    # print(goal.get_higher_y())
    # # choice = random.randint(0,len(goals)-1)
    # # print(choice)
    # # goal = goals[choice]
    # agents.append(Agent(size, mass, pos, goal, desiredSpeed=desiredSpeed))
    # i+=1


    env = Environment(100, walls, goals, agents, {}, instruments, smoke)

    if view:
        viewer = EnvironmentViewer(env)
        viewer.draw()

    env.step()

    # print(env.instruments[0].metric)
    # Run until all agents have escaped
    start = 0
    while env.instruments[0].metric[-1] <= len(env.agents):
        env.step()
        if view:
            viewer.draw()
            # pygame.event.wait()
        if len(env.instruments[0].metric) % 100 == 0:
            if(env.instruments[0].metric[-1] != env.instruments[0].metric[-100]):
                start = 0
            else:
                start+=100
            message = "num escaped: {}, step: {}".format(env.instruments[0].metric[-1], len(env.instruments[0].metric))
            sys.stdout.write('\r' + str(message) + ' ' * 20)
            sys.stdout.write("second since update "+str(start))
            sys.stdout.flush()  # important
            if(start>7000):
                return -1

        if len(env.instruments[0].metric) == 900000000000 or env.instruments[0].metric[-1] == len(env.agents):
            message = "num escaped: {}, step: {}".format(env.instruments[0].metric[-1], len(env.instruments[0].metric))
            sys.stdout.write('\r' + str(message) + ' ' * 20)
            sys.stdout.flush()  # important
            break

    print()
    return env.instruments[0].metric


def runExperiment():
    x = []
    import time

    time_to_escape = []
    #list_test = [20, 50, 100, 200]
    # list_test = [100]
    list_test = [20,50]
    # list_test = [2]
    statistics = -1
    for num_agents in range(len(list_test)):  # (20, 50, 100, 200)
        statistics = -1
        while(statistics==-1):
            statistics = runSimulation(view=False, desiredSpeed=0.6, numAgents=list_test[num_agents], roomHeight=15,
                                   roomWidth=15, doorWidth=1, smoke=False,twoDoors=False, halfMode=False)

        x.append(num_agents)
        time_to_escape.append(len(statistics))
        print(time_to_escape)
    export = [x, time_to_escape]
    # with open("{}.pd".format(datetime.time()), "r") as outfile:
    #     pickle.dump(export, outfile)


if __name__ == '__main__':
    start = time.perf_counter()
    # simResult = runSimulation(view=True, desiredSpeed=1.5, numAgents=10, roomHeight=15, roomWidth=15)
    # print(simResult)
    runExperiment()

    end = time.perf_counter()
    print(f"simulation time: {end - start:0.4f} seconds")

    # thinkplot.plot(defaultExperiment)
    # thinkplot.show()
