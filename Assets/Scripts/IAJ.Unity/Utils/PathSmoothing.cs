using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.IAJ.Unity.Pathfinding.Path;

public class PathSmoothing : MonoBehaviour
{

    public PathSmoothing() { }

    public GlobalPath SmoothPath(GlobalPath currentSolution, Vector3 startPosition)
    {
        GlobalPath smoothPath = new GlobalPath();
		smoothPath.PathPositions.Add(startPosition);
		smoothPath.PathPositions.Add(currentSolution.PathPositions[0]);

		for (int i = 0; i < currentSolution.PathPositions.Count - 2; i++)
		{
			int n = 2;
			while (this.IsWalkable(currentSolution.PathPositions[i], currentSolution.PathPositions[i + n]) && i + n < currentSolution.PathPositions.Count - 1)
			{
				n++;
			}
			smoothPath.PathPositions.Add(currentSolution.PathPositions[i + n - 1]);
			i = i + n - 1;
		}
		smoothPath.PathPositions.Add(currentSolution.PathPositions[currentSolution.PathPositions.Count - 1]);

		return smoothPath;
    }


    private bool IsWalkable(Vector3 p1, Vector3 p2)
    {
        Vector3 direction = p2 - p1;
        if (Physics.Raycast(p1, direction, direction.magnitude))
        {
            return false;
        }
        return true;
    }

}
