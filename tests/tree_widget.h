#pragma once

#include <graphviz/gvc.h>

#include <QPainter>
#include <QPaintEvent>
#include <QString>
#include <QWidget>

class TreeWidget : public QWidget {
 public:
  TreeWidget(const QString& dotFilePath, QWidget* parent = nullptr)
      : QWidget(parent), dotFilePath(dotFilePath) {}

 protected:
  void paintEvent(QPaintEvent* event) override {
    Q_UNUSED(event);

    // Initialize Graphviz context
    GVC_t* gvc = gvContext();
    Agraph_t* graph = agread((char*)dotFilePath.toStdString().c_str(), nullptr);

    // Generate the layout
    gvLayout(gvc, graph, "dot");

    // Draw the graph
    QPainter painter(this);
    drawGraph(&painter, graph);

    // Cleanup
    gvFreeLayout(gvc, graph);
    agclose(graph);
    gvFreeContext(gvc);
  }

 private:
  QString dotFilePath;

  void drawGraph(QPainter* painter, Agraph_t* graph) {
    for (Agnode_t* n = agfstnode(graph); n; n = agnxtnode(graph, n)) {
      // Draw node
      QPointF pos(AGX(n), AGY(n));
      painter->drawEllipse(pos, 10, 10);

      for (Agedge_t* e = agfstout(graph, n); e; e = agnxtout(graph, e)) {
        // Draw edge
        Agnode_t* target = aghead(e);
        QPointF targetPos(AGX(target), AGY(target));
        painter->drawLine(pos, targetPos);
      }
    }
  }

  double AGX(Agnode_t* n) { return atof(agget(n, (char*)"pos")); }

  double AGY(Agnode_t* n) { return atof(agget(n, (char*)"pos") + 1); }
};
