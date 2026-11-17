# Mean-field house thermal coupling over pairwise room graph

Per-room Extended Kalman Filters model passive inter-room drift and active fan Circulation against the scalar difference `(T_house - T_room)` rather than an `N x N` pairwise room-conductance matrix. Pairwise inter-room conductances are collinear and unobservable from room air thermistors alone without floorplan geometry or controlled excitation per room pair, whereas coupling each Thermal Zone to the live House Reference Temperature keeps estimation decoupled (`O(N)`) while giving an identifiable, signed, zero-mean regressor for both passive room-to-house exchange and learned Circulation mixing (`beta_fan`).

> Amended by [0005](0005-oriented-solar-and-room-topology.md): the rejection of dense pairwise conductances stands, but the unweighted whole-house mean it implies as the coupling *reference* does not — a declared adjacency graph now supplies the local reference.
