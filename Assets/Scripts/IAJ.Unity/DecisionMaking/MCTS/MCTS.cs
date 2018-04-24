using Assets.Scripts.GameManager;
using Assets.Scripts.IAJ.Unity.DecisionMaking.GOB;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.DecisionMaking.MCTS
{
    public class MCTS
    {
        public const float C = 1.4f;
        public bool InProgress { get; private set; }
        public int MaxIterations { get; set; }
        public int MaxIterationsProcessedPerFrame { get; set; }
        public int MaxPlayoutDepthReached { get; set; }
        public int MaxSelectionDepthReached { get; private set; }
        public float TotalProcessingTime { get; private set; }
		public float SumProcessingTime { get; set; }
        public MCTSNode BestFirstChild { get; set; }
        public List<GOB.Action> BestActionSequence { get; private set; }


        public int CurrentIterations { get; set; }
        protected int CurrentIterationsInFrame { get; set; }
        protected int CurrentDepth { get; set; }

        protected CurrentStateWorldModel CurrentStateWorldModel { get; set; }
        protected MCTSNode InitialNode { get; set; }
        protected System.Random RandomGenerator { get; set; }



        public MCTS(CurrentStateWorldModel currentStateWorldModel)
        {
            this.InProgress = false;
            this.CurrentStateWorldModel = currentStateWorldModel;
            this.MaxIterations = 100;
            this.MaxIterationsProcessedPerFrame = 10;
            this.RandomGenerator = new System.Random();
			this.SumProcessingTime = 0.0f;
			this.MaxSelectionDepthReached = 0;
			this.MaxPlayoutDepthReached = 0;
		}


        public void InitializeMCTSearch()
        {
            this.CurrentIterations = 0;
            this.CurrentIterationsInFrame = 0;
            this.TotalProcessingTime = 0.0f;
            this.CurrentStateWorldModel.Initialize();
            this.InitialNode = new MCTSNode(this.CurrentStateWorldModel)
            {
                Action = null,
                Parent = null,
                PlayerID = 0
            };
            this.InProgress = true;
            this.BestFirstChild = null;
            this.BestActionSequence = new List<GOB.Action>();
        }


        public GOB.Action Run()
        {
            MCTSNode selectedNode;
            Reward reward;

            var startTime = Time.realtimeSinceStartup;
            this.CurrentIterationsInFrame = 0;
            
            while (this.CurrentIterations < this.MaxIterations) {
                this.CurrentIterations++;
                this.CurrentIterationsInFrame++;
                selectedNode = this.Selection(this.InitialNode);
                reward = this.Playout(selectedNode.State);
                this.Backpropagate(selectedNode, reward);

                if (this.CurrentIterationsInFrame < this.MaxIterationsProcessedPerFrame) {
                    this.CurrentIterationsInFrame++;
                }
                else {
                    this.TotalProcessingTime += Time.realtimeSinceStartup - startTime;
					//this.SumProcessingTime += TotalProcessingTime;
                    return this.BestChild(this.InitialNode).Action;
                }
            }

            this.InProgress = false;

            this.CurrentIterations = 0;
            this.TotalProcessingTime += Time.realtimeSinceStartup - startTime;
			this.SumProcessingTime += TotalProcessingTime;
			this.BestFirstChild = this.BestChild(this.InitialNode);

            return this.GetBestActionSequenceAndAction();
        }


        private MCTSNode Selection(MCTSNode initialNode)
        {
            GOB.Action nextAction;
            MCTSNode currentNode = initialNode;
			int DepthReached = 0;
            while (!currentNode.State.IsTerminal()) {

				DepthReached++;

                nextAction = currentNode.State.GetNextAction();

                if (nextAction != null) {
                    return this.Expand(currentNode, nextAction);
                }
                else { 
                    currentNode = BestUCTChild(currentNode);
                }
            }

			if (DepthReached > this.MaxSelectionDepthReached)
				this.MaxSelectionDepthReached = DepthReached;

            return currentNode;
        }
        

        protected virtual Reward Playout(WorldModel initialPlayoutState)
        {   
            WorldModel childWorldModel = initialPlayoutState.GenerateChildWorldModel();
			GOB.Action[] actions = childWorldModel.GetExecutableActions();
			int DepthReached = 0;
			while (!childWorldModel.IsTerminal()) {

                if (actions.Length > 0) {
					int index = this.RandomGenerator.Next(actions.Length);
					GOB.Action a = actions[index];
					//GOB.Action a = actions[6];
					a.ApplyActionEffects(childWorldModel);
                    childWorldModel.CalculateNextPlayer();
                }
				DepthReached++;
            }

			if (DepthReached > this.MaxPlayoutDepthReached) {
				this.MaxPlayoutDepthReached = DepthReached;
			}

            Reward reward = new Reward
            {
                PlayerID = childWorldModel.GetNextPlayer(),
                Value = childWorldModel.GetScore()
            };
            return reward;
        }

      
        protected virtual void Backpropagate(MCTSNode node, Reward reward)
        {
            while (node != null)
            {
                node.N += 1;
                if (node.Parent == null || reward.PlayerID == node.Parent.PlayerID)
                {
                    node.Q += reward.Value;
                }
                else
                {
                    node.Q -= reward.Value;
                }
                node = node.Parent;
            }
        }


        private MCTSNode Expand(MCTSNode parent, GOB.Action action)
        {
            var childWorldModel = parent.State.GenerateChildWorldModel();
            action.ApplyActionEffects(childWorldModel);
            childWorldModel.CalculateNextPlayer();
            var childNode = new MCTSNode(childWorldModel)
            {
                Action = action,
                PlayerID = parent.PlayerID,
                Parent = parent
            };

            parent.ChildNodes.Add(childNode);

            return childNode;
        }
   
    
        //gets the best child of a node, using the UCT formula
        protected virtual MCTSNode BestUCTChild(MCTSNode node)
        {
            MCTSNode bestChildNode = null;
            float bestUCT = -1.0f;

            foreach (var child in node.ChildNodes) {

                var u = child.Q / child.N;
                var currentUCT = u + C * Mathf.Sqrt(Mathf.Log(child.Parent.N) / child.N);

                if (currentUCT > bestUCT) {
                    bestUCT = currentUCT;
                    bestChildNode = child;
                }
            }
            return bestChildNode;
        }
       
        //this method is very similar to the bestUCTChild, but it is used to return the final action of the MCTS search, and so we do not care about
        //the exploration factor
        private MCTSNode BestChild(MCTSNode node)
        {
            MCTSNode bestChildNode = null;
            float bestUCT = -1.0f;

            foreach (var child in node.ChildNodes) {

                var u = child.Q / child.N;
                var currentUCT = u + Mathf.Sqrt(Mathf.Log(child.Parent.N) / child.N);

                if (currentUCT > bestUCT) {
                    bestUCT = currentUCT;
                    bestChildNode = child;
                }
            }
            return bestChildNode;
        }

        private GOB.Action GetBestActionSequenceAndAction()
        {
            this.BestActionSequence.Clear();
            var bestNode = this.BestFirstChild;

            while (true)
            {
                if (bestNode == null || bestNode.State.IsTerminal())
                {
                    break;
                }
                this.BestActionSequence.Add(bestNode.Action);
                bestNode = BestChild(bestNode);
            }

            if (this.BestFirstChild != null)
            {
                return this.BestFirstChild.Action;
            }
            return null;
        }

    }
}

