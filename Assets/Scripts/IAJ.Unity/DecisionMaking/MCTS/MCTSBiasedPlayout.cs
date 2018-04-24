using Assets.Scripts.GameManager;
using System;
using System.Collections.Generic;
using Assets.Scripts.IAJ.Unity.DecisionMaking.GOB;
using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.DecisionMaking.MCTS
{
    public class MCTSBiasedPlayout : MCTS
    {
        public MCTSBiasedPlayout(CurrentStateWorldModel currentStateWorldModel) : base(currentStateWorldModel)
        {
        }

        protected override Reward Playout(WorldModel initialPlayoutState)
        {
            WorldModel childWorldModel = initialPlayoutState.GenerateChildWorldModel();
			int DepthReached = 0;

			while (!childWorldModel.IsTerminal())
            {
                GOB.Action[] actions = childWorldModel.GetExecutableActions();
                double[] actionIndexes = new double[actions.Length];
                double heuristicValue = 0.0;
                double accumulatedHeuristicValue = 0.0;
                double randomIndex;
                int chosenActionIndex = 0;
                for (int i = 0; i < actions.Length; i++)
                {

                    heuristicValue = actions[i].H(childWorldModel);
                    accumulatedHeuristicValue += Math.Pow(Math.E, -heuristicValue);
                    actionIndexes[i] = accumulatedHeuristicValue;
                }

                randomIndex = this.RandomGenerator.NextDouble() * accumulatedHeuristicValue;
                for (int i = 0; i < actions.Length; i++)
                {
                    if (randomIndex <= actionIndexes[i])
                    {
                        chosenActionIndex = i;
                        break;
                    }

                }
                actions[chosenActionIndex].ApplyActionEffects(childWorldModel);
                childWorldModel.CalculateNextPlayer();
				DepthReached++;
            }

			if (DepthReached > this.MaxPlayoutDepthReached)
			{
				this.MaxPlayoutDepthReached = DepthReached;
			}

			Reward reward = new Reward
            {
                PlayerID = this.InitialNode.PlayerID,
                Value = childWorldModel.GetScore()
            };
            return reward;
        }
    }
}