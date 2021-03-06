//*********************************************************
// Tested classes:
//    - SugiyamaLayout
//
//  Author: Carsten Gutwenger, Tilo Wiedera
//*********************************************************

#include <bandit/bandit.h>

#include <ogdf/layered/SugiyamaLayout.h>
#include <ogdf/layered/FastHierarchyLayout.h>
#include <ogdf/layered/MedianHeuristic.h>
#include <ogdf/layered/BarycenterHeuristic.h>
#include <ogdf/layered/OptimalHierarchyLayout.h>
#include <ogdf/layered/OptimalRanking.h>
#include <ogdf/layered/LongestPathRanking.h>
#include <ogdf/layered/GreedyCycleRemoval.h>
#include <ogdf/layered/DfsAcyclicSubgraph.h>

#include "layout_helpers.h"

using namespace ogdf;

go_bandit([](){ bandit::describe("Sugiyama layouts", [](){
	SugiyamaLayout sugi, sugiOpt, sugiTrans, sugiRuns;

	sugi.setLayout(new FastHierarchyLayout);
	describeLayoutModule("Sugiyama with fast hierarchy", sugi, 0, GR_ALL, 100);

	std::string desc = "Sugiyama with optimal ranking, median";
#ifdef OGDF_LP_SOLVER
	sugiOpt.setLayout(new OptimalHierarchyLayout);
	desc += " and optimal hierarchy";
#endif
	OptimalRanking *optr = new OptimalRanking;
	optr->setSubgraph(new GreedyCycleRemoval);
	sugiOpt.setRanking(optr);
	sugiOpt.setCrossMin(new MedianHeuristic);
	sugiOpt.transpose(false);
	describeLayoutModule(desc.c_str(), sugiOpt, 0, 50);

	sugiTrans.transpose(true);
	describeLayoutModule("Sugiyama with transpositions", sugiTrans, 0, GR_ALL, 100);

	sugiRuns.runs(40);
	describeLayoutModule("Sugiyama with 40 runs", sugiRuns, 0, GR_ALL, 50);
}); });
