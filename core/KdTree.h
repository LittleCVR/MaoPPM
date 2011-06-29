/*
 * =====================================================================================
 *
 *       Filename:  KdTree.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-29 00:54:18
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MAOPPM_CORE_KD_TREE_H
#define MAOPPM_CORE_KD_TREE_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files from std C/C++
 *----------------------------------------------------------------------------*/
#include    <algorithm>

/*---------------------------------------------------------------------------
 *  header files of our own
 *---------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {

class KdTree {
    public:
        enum Flag {
            Null      = 0,
            Leaf      = 1 << 0,
            AxisX     = 1 << 1,
            AxisY     = 1 << 2,
            AxisZ     = 1 << 3,
            User      = 1 << 4
        };

    public:
        template<typename T>
        static bool positionXComparator(const T & node1, const T & node2)
        {
            return node1.position.x < node2.position.x;
        }
        template<typename T>
        static bool positionYComparator(const T & node1, const T & node2)
        {
            return node1.position.y < node2.position.y;
        }
        template<typename T>
        static bool positionZComparator(const T & node1, const T & node2)
        {
            return node1.position.z < node2.position.z;
        }

    public:
        template<typename T>
        static void build(T * nodeList,
                unsigned int start, unsigned int end, T * tree,
                unsigned int root, optix::float3 bbMin, optix::float3 bbMax)
        {
            if (end - start == 0) {
                // Make a fake node.
                tree[root].flags = Null;
                tree[root].flux  = optix::make_float3(0.0f);
                return;
            }
            if (end - start == 1) {
                // Create a leaf node.
                nodeList[start].flags |= Leaf;
                tree[root] = nodeList[start];
                return;
            }

            // Choose the longest axis.
            unsigned int axis = 0;
            optix::float3 bbDiff = bbMax - bbMin;
            if (bbDiff.x > bbDiff.y) {
                if (bbDiff.x > bbDiff.z)
                    axis = AxisX;
                else
                    axis = AxisZ;
            } else {
                if (bbDiff.y > bbDiff.z)
                    axis = AxisY;
                else
                    axis = AxisZ;
            }

            // Partition the node list.
            unsigned int median = (start + end) / 2;
            if (axis == AxisX)
                nth_element(&nodeList[start], &nodeList[median], &nodeList[end], positionXComparator<T>);
            else if (axis == AxisY)
                nth_element(&nodeList[start], &nodeList[median], &nodeList[end], positionYComparator<T>);
            else if (axis == AxisZ)
                nth_element(&nodeList[start], &nodeList[median], &nodeList[end], positionZComparator<T>);
            else
                assert(false);  // This should never happen.
            nodeList[median].flags |= axis;
            tree[root] = nodeList[median];

            // Calculate new bounding box.
            optix::float3 leftMax  = bbMax;
            optix::float3 rightMin = bbMin;
            optix::float3 midPoint = tree[root].position;
            switch (axis) {
                case AxisX:
                    rightMin.x = midPoint.x;
                    leftMax.x  = midPoint.x;
                    break;
                case AxisY:
                    rightMin.y = midPoint.y;
                    leftMax.y  = midPoint.y;
                    break;
                case AxisZ:
                    rightMin.z = midPoint.z;
                    leftMax.z  = midPoint.z;
                    break;
                default:
                    assert(false);
                    break;
            }

            build(nodeList, start, median, tree, 2*root+1, bbMin,  leftMax);
            build(nodeList, median+1, end, tree, 2*root+2, rightMin, bbMax);
        }   /* -----  end of method KdTree::build----- */

#ifdef __CUDACC__
    public:
        template <typename T, typename G>
        __device__ __inline__ static void find(
                const optix::float3 & point, const T * nodeList,
                G * gatherer, float * maxDistanceSquared)
        {
            optix::float3 flux = optix::make_float3(0.0f);

            uint stack[32];
            uint stackPosition = 0;
            uint stackNode     = 0;
            stack[stackPosition++] = 0;
            do {
                const T * node = &nodeList[stackNode];
                if (node->flags == Null)
                    stackNode = stack[--stackPosition];
                else {
                    float3 diff = point - node->position;
                    float distanceSquared = optix::dot(diff, diff);
                    if (distanceSquared < *maxDistanceSquared)
                        gatherer->gather(point, node, distanceSquared, maxDistanceSquared);

                    if (node->flags & Leaf)
                        stackNode = stack[--stackPosition];
                    else {
                        float d;
                        if      (node->flags & AxisX)  d = diff.x;
                        else if (node->flags & AxisY)  d = diff.y;
                        else                           d = diff.z;

                        // Calculate the next child selector. 0 is left, 1 is right.
                        int selector = d < 0.0f ? 0 : 1;
                        if (d*d < *maxDistanceSquared)
                            stack[stackPosition++] = (stackNode << 1) + 2 - selector;
                        stackNode = (stackNode << 1) + 1 + selector;
                    }
                }
            } while (stackNode != 0) ;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};  /* -----  end of class KdTree  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_KD_TREE_H  ----- */
