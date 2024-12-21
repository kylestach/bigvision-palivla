from scalax.sharding import MeshShardingHelper, ShardingRule, PartitionSpec


class ShardingMetadata:
    mesh: MeshShardingHelper
    model_sharding_rule: ShardingRule | PartitionSpec

    def __init__(
        self,
        mesh: MeshShardingHelper,
        model_sharding_rule: ShardingRule | PartitionSpec,
    ):
        self.mesh = mesh
        self.model_sharding_rule = model_sharding_rule
