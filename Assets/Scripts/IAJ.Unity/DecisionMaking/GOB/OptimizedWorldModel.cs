using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.GameManager;

namespace Assets.Scripts.IAJ.Unity.DecisionMaking.GOB
{
    public class OptimizedWorldModel
    {
        
        private List<Action> Actions { get; set; }
        protected IEnumerator<Action> ActionEnumerator { get; set; }

        //private Dictionary<string, float> GoalValues { get; set; }

        protected OptimizedWorldModel Parent { get; set; }

        private const int WORLD_SIZE = 22;
        public object[] WorldState = new object[WORLD_SIZE];
        //ARRAY POSITIONS

        private const int MANA = 0;
        private const int HP = 1;
        private const int MAXHP = 2;
        private const int XP = 3;
        private const int TIME = 4;
        private const int MONEY = 5;
        private const int LEVEL = 6;
        private const int POSITION = 7;
        private const int CHEST_1 = 8;
        private const int CHEST_2 = 9;
        private const int CHEST_3 = 10;
        private const int CHEST_4 = 11;
        private const int CHEST_5 = 12;
        private const int MANA_1 = 13;
        private const int MANA_2 = 14;
        private const int HEALTH_1 = 15;
        private const int HEALTH_2 = 16;
        private const int SKELETON_1 = 17;
        private const int SKELETON_2 = 18;
        private const int ORC_1 = 19;
        private const int ORC_2 = 20;
        private const int DRAGON = 21;

        //TAGS

        private const string MANA_TAG_1 = "ManaPotion1";
        private const string MANA_TAG_2 = "ManaPotion2";
        private const string HEALTH_TAG_1 = "HealthPotion1";
        private const string HEALTH_TAG_2 = "HealthPotion2";
        private const string CHEST_TAG_1 = "Chest1";
        private const string CHEST_TAG_2 = "Chest2";
        private const string CHEST_TAG_3 = "Chest3";
        private const string CHEST_TAG_4 = "Chest4";
        private const string CHEST_TAG_5 = "Chest5";
        private const string SKELETON_TAG_1 = "Skeleton1";
        private const string SKELETON_TAG_2 = "Skeleton2";
        private const string ORC_TAG_1 = "Orc1";
        private const string ORC_TAG_2 = "Orc2";
        private const string DRAGON_TAG = "Dragon";
        


        public OptimizedWorldModel(GameManager.GameManager m, List<Action> actions)
        {

            //this.GoalValues = new Dictionary<string, float>();

            //create information for array
            this.WorldState[MANA] = m.characterData.Mana;
            this.WorldState[HP] = m.characterData.HP;
            this.WorldState[MAXHP] = m.characterData.MaxHP;
            this.WorldState[XP] = m.characterData.XP;
            this.WorldState[TIME] = m.characterData.Time;
            this.WorldState[MONEY] = m.characterData.Money;
            this.WorldState[LEVEL] = m.characterData.Level;
            this.WorldState[POSITION] = m.characterData.CharacterGameObject.transform.position;

            for (var i = 8; i < WORLD_SIZE; i++)
            {
                this.WorldState[i] = true;
            }

            this.Actions = actions;
            this.ActionEnumerator = actions.GetEnumerator();
        }

        public OptimizedWorldModel(OptimizedWorldModel parent)
        {
            //this.GoalValues = new Dictionary<string, float>();
            //copy information for array
            for (var i = 0; i < WORLD_SIZE; i++)
            {
                this.WorldState[i] = parent.WorldState[i];
            }
            
            this.Actions = parent.Actions;
            this.Parent = parent;
            this.ActionEnumerator = this.Actions.GetEnumerator();
        }

        public virtual object GetProperty(string propertyName)
        {
            object prop = null;

            if (Properties.MANA == propertyName)
            {
                prop = this.WorldState[MANA];
            }

            else if (Properties.HP == propertyName)
            {
                prop = this.WorldState[HP];
            }

            else if (Properties.MAXHP == propertyName)
            {
                prop = this.WorldState[MAXHP];
            }

            else if (Properties.XP == propertyName)
            {
                prop = this.WorldState[XP];
            }

            else if (Properties.TIME == propertyName)
            {
                prop = this.WorldState[TIME];
            }

            else if (Properties.MONEY == propertyName)
            {
                prop = this.WorldState[MONEY];
            }

            else if (Properties.LEVEL == propertyName)
            {
                prop = this.WorldState[LEVEL];
            }

            else if (Properties.POSITION == propertyName)
            {
                prop = this.WorldState[POSITION];
            }

            else if (MANA_TAG_1 == propertyName)
            {
                prop = this.WorldState[MANA_1];
            }

            else if (MANA_TAG_2 == propertyName)
            {
                prop = this.WorldState[MANA_2];
            }

            else if (HEALTH_TAG_1 == propertyName)
            {
                prop = this.WorldState[HEALTH_1];
            }

            else if (HEALTH_TAG_2 == propertyName)
            {
                prop = this.WorldState[HEALTH_2];
            }

            else if (CHEST_TAG_1 == propertyName)
            {
                prop = this.WorldState[CHEST_1];
            }

            else if (CHEST_TAG_2 == propertyName)
            {
                prop = this.WorldState[CHEST_2];
            }

            else if (CHEST_TAG_3 == propertyName)
            {
                prop = this.WorldState[CHEST_3];
            }

            else if (CHEST_TAG_4 == propertyName)
            {
                prop = this.WorldState[CHEST_4];
            }

            else if (CHEST_TAG_5 == propertyName)
            {
                prop = this.WorldState[CHEST_5];
            }

            else if (SKELETON_TAG_1 == propertyName)
            {
                prop = this.WorldState[SKELETON_1];
            }

            else if (SKELETON_TAG_2 == propertyName)
            {
                prop = this.WorldState[SKELETON_2];
            }

            else if (ORC_TAG_1 == propertyName)
            {
                prop = this.WorldState[ORC_1];
            }

            else if (ORC_TAG_2 == propertyName)
            {
                prop = this.WorldState[ORC_2];
            }

            else if (DRAGON_TAG == propertyName)
            {
                prop = this.WorldState[DRAGON];
            }

            return prop;
        }

        public virtual void SetProperty(string propertyName, object value)
        {
            if (Properties.MANA == propertyName)
            {
                this.WorldState[MANA] = value;
            }

            else if (Properties.HP == propertyName)
            {
                this.WorldState[HP] = value;
            }

            else if (Properties.MAXHP == propertyName)
            {
                this.WorldState[MAXHP] = value;
            }

            else if (Properties.XP == propertyName)
            {
                this.WorldState[XP] = value;
            }

            else if (Properties.TIME == propertyName)
            {
                this.WorldState[TIME] = value;
            }

            else if (Properties.MONEY == propertyName)
            {
                this.WorldState[MONEY] = value;
            }

            else if (Properties.LEVEL == propertyName)
            {
                this.WorldState[LEVEL] = value;
            }

            else if (Properties.POSITION == propertyName)
            {
                this.WorldState[POSITION] = value;
            }

            else if (MANA_TAG_1 == propertyName)
            {
                this.WorldState[MANA_1] = value;
            }

            else if (MANA_TAG_2 == propertyName)
            {
                this.WorldState[MANA_2] = value;
            }

            else if (HEALTH_TAG_1 == propertyName)
            {
                this.WorldState[HEALTH_1] = value;
            }

            else if (HEALTH_TAG_2 == propertyName)
            {
                this.WorldState[HEALTH_2] = value;
            }

            else if (CHEST_TAG_1 == propertyName)
            {
                this.WorldState[CHEST_1] = value;
            }

            else if (CHEST_TAG_2 == propertyName)
            {
                this.WorldState[CHEST_2] = value;
            }

            else if (CHEST_TAG_3 == propertyName)
            {
                this.WorldState[CHEST_3] = value;
            }

            else if (CHEST_TAG_4 == propertyName)
            {
                this.WorldState[CHEST_4] = value;
            }

            else if (CHEST_TAG_5 == propertyName)
            {
                this.WorldState[CHEST_5] = value;
            }

            else if (SKELETON_TAG_1 == propertyName)
            {
                this.WorldState[SKELETON_1] = value;
            }

            else if (SKELETON_TAG_2 == propertyName)
            {
                this.WorldState[SKELETON_2] = value;
            }

            else if (ORC_TAG_1 == propertyName)
            {
                this.WorldState[ORC_1] = value;
            }

            else if (ORC_TAG_2 == propertyName)
            {
                this.WorldState[ORC_2] = value;
            }

            else if (DRAGON_TAG == propertyName)
            {
                this.WorldState[DRAGON] = value;
            }
        }

        /*public virtual float GetGoalValue(string goalName)
        {
            //recursive implementation of WorldModel
            if (this.GoalValues.ContainsKey(goalName))
            {
                return this.GoalValues[goalName];
            }
            else if (this.Parent != null)
            {
                return this.Parent.GetGoalValue(goalName);
            }
            else
            {
                return 0;
            }
        }

        public virtual void SetGoalValue(string goalName, float value)
        {
            var limitedValue = value;
            if (value > 10.0f)
            {
                limitedValue = 10.0f;
            }

            else if (value < 0.0f)
            {
                limitedValue = 0.0f;
            }

            this.GoalValues[goalName] = limitedValue;
        }*/

        public virtual OptimizedWorldModel GenerateChildWorldModel()
        {
            return new OptimizedWorldModel(this);
        }

        /*public float CalculateDiscontentment(List<Goal> goals)
        {
            var discontentment = 0.0f;

            foreach (var goal in goals)
            {
                var newValue = this.GetGoalValue(goal.Name);

                discontentment += goal.GetDiscontentment(newValue);
            }

            return discontentment;
        }*/

        /*public virtual Action GetNextAction()
        {
            Action action = null;
            //returns the next action that can be executed or null if no more executable actions exist
            if (this.ActionEnumerator.MoveNext())
            {
                action = this.ActionEnumerator.Current;
            }

            while (action != null && !action.CanExecute(this))
            {
                if (this.ActionEnumerator.MoveNext())
                {
                    action = this.ActionEnumerator.Current;
                }
                else
                {
                    action = null;
                }
            }

            return action;
        }

        public virtual Action[] GetExecutableActions()
        {
            return this.Actions.Where(a => a.CanExecute(this)).ToArray();
        }*/

        public virtual bool IsTerminal()
        {
            return true;
        }


        public virtual float GetScore()
        {
            return 0.0f;
        }

        public virtual int GetNextPlayer()
        {
            return 0;
        }

        public virtual void CalculateNextPlayer()
        {
        }
    }
}
