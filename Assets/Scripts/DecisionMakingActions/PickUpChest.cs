using Assets.Scripts.GameManager;
using Assets.Scripts.IAJ.Unity.DecisionMaking.GOB;
using UnityEngine;

namespace Assets.Scripts.DecisionMakingActions
{
    public class PickUpChest : WalkToTargetAndExecuteAction
    {

        public PickUpChest(AutonomousCharacter character, GameObject target) : base("PickUpChest",character,target)
        {
        }


        public override float GetGoalChange(Goal goal)
        {
            var change = base.GetGoalChange(goal);
            if (goal.Name == AutonomousCharacter.GET_RICH_GOAL) change -= 5.0f;
            return change;
        }

        public override bool CanExecute()
        {
            if (!base.CanExecute()) return false;
            return true;
        }

        public override bool CanExecute(WorldModel worldModel)
        {
            if (!base.CanExecute(worldModel)) return false;
            return true;
        }

        public override void Execute()
        {
            base.Execute();
            this.Character.GameManager.PickUpChest(this.Target);
        }

        public override void ApplyActionEffects(WorldModel worldModel)
        {
            base.ApplyActionEffects(worldModel);

            var goalValue = worldModel.GetGoalValue(AutonomousCharacter.GET_RICH_GOAL);
            worldModel.SetGoalValue(AutonomousCharacter.GET_RICH_GOAL, goalValue - 5.0f);

            var money = (int)worldModel.GetProperty(Properties.MONEY);
            worldModel.SetProperty(Properties.MONEY, money + 5);

            //disables the target object so that it can't be reused again
            worldModel.SetProperty(this.Target.name, false);
        }

        public override float H(WorldModel currentWorldModel)
        {
            var hp = (int)currentWorldModel.GetProperty(Properties.HP);
            var lvl = (int)currentWorldModel.GetProperty(Properties.LEVEL);

            var isDragonAlive = (bool)currentWorldModel.GetProperty("Dragon");
            var orc1Alive = (bool)currentWorldModel.GetProperty("Orc1");
            var orc2Alive = (bool)currentWorldModel.GetProperty("Orc2");

			if (this.Target.name.Equals("Chest5") && isDragonAlive && hp <= 20)
			{
				return 100.0f;
			}

			if (this.Target.name.Equals("Chest3") && orc1Alive && hp <= 10)
			{
				return 100.0f;
			}

			if (this.Target.name.Equals("Chest2") && orc2Alive && hp <= 10)
			{
				return 100.0f;
			}

			if (hp <= 5)
				return 10.0f;

			if (lvl == 3)
				return 0.0f;

			return 0.5f;
        }
    }
}
