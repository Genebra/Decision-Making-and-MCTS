﻿using Assets.Scripts.GameManager;
using Assets.Scripts.IAJ.Unity.DecisionMaking.GOB;
using System;
using UnityEngine;

namespace Assets.Scripts.DecisionMakingActions
{
	public class GetHealthPotion : WalkToTargetAndExecuteAction
	{
		public GetHealthPotion(AutonomousCharacter character, GameObject target) : base("GetHealthPotion", character, target)
		{
		}

		public override bool CanExecute()
		{
			if (!base.CanExecute()) return false;
			return this.Character.GameManager.characterData.HP < this.Character.GameManager.characterData.MaxHP;
		}

		public override bool CanExecute(WorldModel worldModel)
		{
			if (!base.CanExecute(worldModel)) return false;

			var mana = (int)worldModel.GetProperty(Properties.HP);
			return mana < (int)worldModel.GetProperty(Properties.MAXHP);
		}

		public override void Execute()
		{
			base.Execute();
			this.Character.GameManager.GetHealthPotion(this.Target);
		}

		public override void ApplyActionEffects(WorldModel worldModel)
		{
			base.ApplyActionEffects(worldModel);
			worldModel.SetProperty(Properties.HP, (int)worldModel.GetProperty(Properties.MAXHP));
			worldModel.SetProperty(this.Target.name, false);
		}

		public override float H(WorldModel currentWorldModel)
		{
			var hp = (int)currentWorldModel.GetProperty(Properties.HP);
            var lvl = (int)currentWorldModel.GetProperty(Properties.LEVEL);
			

			if (hp <= 5.0f)
				return 0.0f;
			if (hp <= 10)
				return 1.0f;
			if (lvl == 3)
				return 100.0f;
			else
				return 5.0f;
		}
	}
}
