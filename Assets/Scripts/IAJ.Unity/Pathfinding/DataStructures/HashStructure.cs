using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Pathfinding.DataStructures
{
    public class HashStructure : IClosedSet
    {
        private Dictionary<Vector3, NodeRecord> NodeRecords { get; set; }

        public HashStructure()
        {
            this.NodeRecords = new Dictionary<Vector3, NodeRecord>();
        }

        public void AddToClosed(NodeRecord nodeRecord)
        {
            this.NodeRecords.Add(nodeRecord.node.Position, nodeRecord);
        }

        public ICollection<NodeRecord> All()
        {
            return this.NodeRecords.Values.ToList();
        }

        public void Initialize()
        {
            this.NodeRecords.Clear();
        }

        public void RemoveFromClosed(NodeRecord nodeRecord)
        {
            this.NodeRecords.Remove(nodeRecord.node.Position);
        }

        public NodeRecord SearchInClosed(NodeRecord nodeRecord)
        {
            NodeRecord nodeAux;
            bool nodePresent = this.NodeRecords.TryGetValue(nodeRecord.node.Position, out nodeAux);
            if (nodePresent)
                return nodeAux;
            return null;
        }


    };

}
