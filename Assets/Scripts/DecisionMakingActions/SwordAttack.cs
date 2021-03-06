﻿using Assets.Scripts.GameManager;
using Assets.Scripts.IAJ.Unity.DecisionMaking.GOB;
using UnityEngine;

namespace Assets.Scripts.DecisionMakingActions
{
    public class SwordAttack : WalkToTargetAndExecuteAction
    {
        private int hpChange;
        private int xpChange;

        public SwordAttack(AutonomousCharacter character, GameObject target) : base("SwordAttack",character,target)
        {
           
            if (target.tag.Equals("Skeleton"))
            {
                this.hpChange = -5;
                this.xpChange = 5;
            }
            else if (target.tag.Equals("Orc"))
            {
                this.hpChange = -10;
                this.xpChange = 10;
            }
            else if (target.tag.Equals("Dragon"))
            {
                this.hpChange = -20;
                this.xpChange = 20;
            }
        }

        public override float GetGoalChange(Goal goal)
        {
            var change = base.GetGoalChange(goal);

            if (goal.Name == AutonomousCharacter.SURVIVE_GOAL)
            {
                change += -this.hpChange;
            }
            else if (goal.Name == AutonomousCharacter.GAIN_XP_GOAL)
            {
                change += -this.xpChange;
            }
            
            return change;
        }


        public override void Execute()
        {
            base.Execute();
            this.Character.GameManager.SwordAttack(this.Target);
			//this.Character.GameManager.LevelUp();
        }

        public override void ApplyActionEffects(WorldModel worldModel)
        {
            base.ApplyActionEffects(worldModel);

            var xpValue = worldModel.GetGoalValue(AutonomousCharacter.GAIN_XP_GOAL);
            worldModel.SetGoalValue(AutonomousCharacter.GAIN_XP_GOAL,xpValue-this.xpChange); 

            var surviveValue = worldModel.GetGoalValue(AutonomousCharacter.SURVIVE_GOAL);
            worldModel.SetGoalValue(AutonomousCharacter.SURVIVE_GOAL,surviveValue-this.hpChange);

            var hp = (int)worldModel.GetProperty(Properties.HP);
            worldModel.SetProperty(Properties.HP,hp + this.hpChange);
            var xp = (int)worldModel.GetProperty(Properties.XP);
            worldModel.SetProperty(Properties.XP, xp + this.xpChange);

			this.Character.GameManager.LevelUp();

            //disables the target object so that it can't be reused again
            worldModel.SetProperty(this.Target.name,false);
        }


		public override float H(WorldModel currentWorldModel)
		{
			var hp = (int)currentWorldModel.GetProperty(Properties.HP);

			if (Target.tag.Equals("Dragon"))
			{
				if (hp > 20)
					return 0.0f;
				else
					return 100.0f;
			}
			else if (Target.tag.Equals("Orc"))
			{
				if (hp > 10)
					return 0.5f;
				else
					return 100.0f;
			}
			else if (Target.tag.Equals("Skeleton"))
			{
				if (hp > 5)
				{
					return 0.5f;
				}
				else
				{
					return 100.0f;
				}
			}
			else
				return 0.0f;
		}
    }
}
