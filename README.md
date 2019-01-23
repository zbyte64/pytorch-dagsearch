
ENAS: https://arxiv.org/pdf/1802.03268.pdf
PNN: https://arxiv.org/pdf/1706.03256.pdf


Graph would be a `nn.Sequential` model that allows for skip connections to be registered
```
g = Graph(
  Conv2D(...)
  MaxPool2D(...)
  Conv2D(...)
  )

#Generates a connector for different layer sizes
g.add_link(g.layers[0], g.layers[2])
#Mutes an input, useful for ENAS
g.toggle_link(g.layers[0], g.layers[2])
```
Graphs are also expandable, new nodes are added in front.

`Node` is a special layer that allows for on-the-fly hyperparam search.
Unlike a layer, it specifies it's in and out dim size and initializes candidate cells to that size.
`Connector` is responsible for reshaping and learning identity mappings across layers.
`World` implements the game logic and actions available to the ENAS agent.
`DagSearchEnv` is the Gym environment.
