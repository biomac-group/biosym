from biosym.ocp import collocation
ocp = collocation.Collocation("dorschky_demo/configs/standing2d.yaml")
ocp.solve()
