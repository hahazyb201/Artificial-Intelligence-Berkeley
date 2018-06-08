# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        print(bestScore)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

       


        "*** YOUR CODE HERE ***"
        nearestDistance=None
        nearestFood=None
        for food in currentGameState.getFood().asList():
          dis=manhattanDistance(food, currentGameState.getPacmanPosition())
          if nearestDistance==None or nearestDistance>dis:
            nearestDistance=dis
            nearestFood=food

        disPoint=3.0/(float(manhattanDistance(nearestFood,newPos))+1.0)

        ghostPoint=0
        for ghost in newGhostStates:
          dis=manhattanDistance(ghost.getPosition(), newPos)
          if dis<3:
            ghostPoint=-100000

        eat=0
        if manhattanDistance(nearestFood,newPos)==0:
          eat=200
        if Directions.STOP in action:  
            return float("-inf")  
        if successorGameState.isWin():
            return float("inf") 

        return successorGameState.getScore()+4*disPoint+ghostPoint+eat

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numAgents=gameState.getNumAgents()
        actions=gameState.getLegalActions(0)
        successors=[gameState.generateSuccessor(0,action) for action in actions]
        if(self.depth==0 or gameState.isLose() or gameState.isWin()):
          return self.evaluationFunction(gameState)
        results=[self.minimizer(1,successor,0) for successor in successors]
        maxScore=max(results)
        bestIndices=[i for i in range(len(results)) if results[i]==maxScore]
        return actions[random.choice(bestIndices)]

    def minimizer(self,agentNo,gameState,depth):
      if(gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
      successors=[gameState.generateSuccessor(agentNo,action) for action in gameState.getLegalActions(agentNo)]
      if agentNo<gameState.getNumAgents()-1:
        results=[self.minimizer(agentNo+1,successor,depth) for successor in successors]
      else:
        results=[self.maximizer(0,successor,depth+1) for successor in successors]
      return min(results)

    def maximizer(self,agentNo,gameState,depth):
      if(self.depth==depth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
      successors=[gameState.generateSuccessor(agentNo,action) for action in gameState.getLegalActions(agentNo)]
      results=[self.minimizer(agentNo+1,successor,depth) for successor in successors]
      return max(results)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents=gameState.getNumAgents()
        actions=gameState.getLegalActions(0)
        if(self.depth==0 or gameState.isLose() or gameState.isWin()):
          return self.evaluationFunction(gameState)
        alpha=float("-inf")
        beta=float("inf")
        v=float("-inf")
        i=0
        move=0
        for action in actions:
          successor=gameState.generateSuccessor(0,action)
          res=self.minimizer(1,successor,0,alpha,beta)
          if v<res:
            move=i
            v=res
          if v>beta:
            return actions[move]
          i+=1
          alpha=max(alpha,v)

        return actions[move]

    def minimizer(self,agentNo,gameState,depth,alpha,beta):
      if(self.depth==depth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
      actions=gameState.getLegalActions(agentNo)
      v=float("inf")
      if agentNo<gameState.getNumAgents()-1:
        for action in actions:
          successor=gameState.generateSuccessor(agentNo,action)
          v=min(v,self.minimizer(agentNo+1,successor,depth,alpha,beta))
          if v<alpha:
            return v
          else:
            beta=min(beta,v)
      else:
        for action in actions:
          successor=gameState.generateSuccessor(agentNo,action)
          v=min(v,self.maximizer(0,successor,depth+1,alpha,beta))
          if v<alpha:
            return v
          else:
            beta=min(beta,v)
        
      return v

    def maximizer(self,agentNo,gameState,depth,alpha,beta):
      if(self.depth==depth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
      actions=gameState.getLegalActions(agentNo)
      v=float("-inf")
      for action in actions:
        successor=gameState.generateSuccessor(agentNo,action)
        v=max(v,self.minimizer(agentNo+1,successor,depth,alpha,beta))
        if v>beta:
          return v
        alpha=max(alpha,v)
      return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents=gameState.getNumAgents()
        actions=gameState.getLegalActions(0)
        successors=[gameState.generateSuccessor(0,action) for action in actions]
        if(self.depth==0 or gameState.isLose() or gameState.isWin()):
          return self.evaluationFunction(gameState)
        results=[self.uniformlyChoose(1,successor,0) for successor in successors]
        maxScore=max(results)
        bestIndices=[i for i in range(len(results)) if results[i]==maxScore]
        return actions[random.choice(bestIndices)]

    def uniformlyChoose(self,agentNo,gameState,depth):
      if(self.depth==0 or gameState.isLose() or gameState.isWin()):
          return float(self.evaluationFunction(gameState))
      successors=[gameState.generateSuccessor(agentNo,action) for action in gameState.getLegalActions(agentNo)]
      if agentNo<gameState.getNumAgents()-1:
        results=[self.uniformlyChoose(agentNo+1,successor,depth) for successor in successors]
      else:
        results=[self.maximizer(0,successor,depth+1) for successor in successors]
      return float(sum(results))/float(len(results))

    def maximizer(self,agentNo,gameState,depth):
      if(self.depth==depth or gameState.isLose() or gameState.isWin()):
        return float(self.evaluationFunction(gameState))
      successors=[gameState.generateSuccessor(agentNo,action) for action in gameState.getLegalActions(agentNo)]
      results=[self.uniformlyChoose(agentNo+1,successor,depth) for successor in successors]
      return max(results) 

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghostPoint=0
    foodPoint=0
    safePoint=0

    pacPos= currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    minFoodDis=10000000
    for food in foods:
      dis=bfsDis(pacPos,food,currentGameState.getWalls())
      minFoodDis=min(minFoodDis,dis)
    if minFoodDis==10000000:
      minFoodDis=0
    foodPoint=-0.5*minFoodDis

    if len(ghosts)==0:
      return foodPoint+safePoint

    

    for g in ghosts:
      if g.scaredTimer <= 0:
        safePoint-=5
        pgDis=bfsDis(pacPos,g.getPosition(),currentGameState.getWalls())
        if pgDis<5:
          ghostPoint-=100/(float(pgDis)+0.1)

    return float(currentGameState.getScore())+float(foodPoint)+float(safePoint)+ghostPoint

def bfsDis(p,g,walls):
  if p==g:
    return 0
  toVisit=util.Queue()
  toVisit.push((p,0))
  visited=set()
  visited.add(p)
  while not toVisit.isEmpty():
    pos,dis=toVisit.pop()
    if pos==g:
      return dis
    successors=getFringe(pos,walls)
    for newPos in successors:
      if(newPos in visited):
        continue
      toVisit.push((newPos,dis+1))
      visited.add(newPos)
  print "hh"

def getFringe(pos,walls):
  successors=[]
  if pos[0]>0:
    if not walls[pos[0]-1][pos[1]]:
      successors.append((pos[0]-1,pos[1]))
  if pos[0]<walls.width-1:
    if not walls[pos[0]+1][pos[1]]:
      successors.append((pos[0]+1,pos[1]))
  if pos[1]>0:
    if not walls[pos[0]][pos[1]-1]:
      successors.append((pos[0],pos[1]-1))
  if pos[1]<walls.height-1:
    if not walls[pos[0]][pos[1]+1]:
      successors.append((pos[0],pos[1]+1))

  return successors



# Abbreviation
better = betterEvaluationFunction

