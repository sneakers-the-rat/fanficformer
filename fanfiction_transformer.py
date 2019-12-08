from tensor2tensor.models.transformer import Transformer
from tensor2tensor.utils import registry


@registry.register_model
class Fanfiction_Transformer(Transformer):
    """
    Fanfiction transformer - transformer with additional

    * Conditional generation
    * Embedding for condition
    """


    def body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs. [batch_size, input_length, 1,
                hidden_dim].
              "labels": Categories of transformer inputs - ie. class of image, or
                in this case the universe the fanfiction story comes from
              "targets": Target decoder outputs. [batch_size, decoder_length, 1,
                hidden_dim]
              "target_space_id": A scalar int from data_generators.problem.SpaceID.

        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        hparams = self._hparams

        losses = []

        if self.has_input:
            inputs = self._prepare_inputs_for_body(features)
            target_space = features["target_space_id"]
            encoder_output, encoder_decoder_attention_bias = self.encode(
              inputs, target_space, hparams, features=features, losses=losses)
        else:
            encoder_output, encoder_decoder_attention_bias = (None, None)

        targets = features["targets"]
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
            targets, hparams, features=features)

        # a la image_transformer.py
        if not hparams.unconditional:
            labels = features["labels"]
            decoder_input += tf.reshape(
                  labels,
                  [common_layers.shape_list(targets)[0], 1, 1, hparams.hidden_size])


        # Not all subclasses of Transformer support keyword arguments related to
        # recurrent memory, so only pass these arguments if memory is enabled.
        decode_kwargs = {}
        if self.recurrent_memory_by_layer is not None:
            # TODO(kitaev): The chunk_number feature currently has the same shape as
            # "targets", but this is only for the purposes of sharing sharding code.
            # In fact every token within an example must have the same chunk number.
            chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
            chunk_number_each_example = chunk_number_each_token[:, 0]
            # Uncomment the code below to verify that tokens within a batch share the
            # same chunk number:
            # with tf.control_dependencies([
            #     tf.assert_equal(chunk_number_each_token,
            #                     chunk_number_each_example[:, None])
            # ]):
            #   chunk_number_each_example = tf.identity(chunk_number_each_example)
            decode_kwargs = dict(
              recurrent_memory_by_layer=self.recurrent_memory_by_layer,
              chunk_number=chunk_number_each_example,
              )
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, "targets"),
            losses=losses,
            **decode_kwargs
            )
        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
              expected_attentions, self.attention_weights,
              hparams.expected_attention_loss_type,
              hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        ret = tf.reshape(decoder_output, targets_shape)
        if losses:
            return ret, {"extra_loss": tf.add_n(losses)}
        else:
            return ret


@registry.register_hparams
def fanficformer_base():
    hparams = common_hparams.basic_params1()
    hparams.norm_type = "layer"
    hparams.hidden_size = 512
    hparams.batch_size = 1024
    hparams.max_length = 1024
    hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
    hparams.optimizer_adam_epsilon = 1e-9
    hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
    hparams.learning_rate_constant = 2.0
    hparams.learning_rate_decay_scheme = "noam"
    hparams.learning_rate = 0.2
    hparams.learning_rate_warmup_steps = 8000
    hparams.initializer_gain = 1.0
    hparams.num_hidden_layers = 6
    hparams.initializer = "uniform_unit_scaling"
    hparams.weight_decay = 0.0
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.997
    hparams.num_sampled_classes = 0
    hparams.label_smoothing = 0.1
    hparams.shared_embedding_and_softmax_weights = True
    hparams.symbol_modality_num_shards = 16

    # Add new ones like this.
    hparams.add_hparam("filter_size", 2048)
    # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
    hparams.add_hparam("num_encoder_layers", 0)
    hparams.add_hparam("num_decoder_layers", 0)
    # Attention-related flags.
    hparams.add_hparam("num_heads", 8)
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("ffn_layer", "dense_relu_dense")
    hparams.add_hparam("parameter_attention_key_channels", 0)
    hparams.add_hparam("parameter_attention_value_channels", 0)
    # All hyperparameters ending in "dropout" are automatically set to 0.0
    # when not in training mode.
    hparams.add_hparam("attention_dropout", 0.1)
    hparams.add_hparam("attention_dropout_broadcast_dims", "")
    hparams.add_hparam("relu_dropout", 0.1)
    hparams.add_hparam("relu_dropout_broadcast_dims", "")
    hparams.add_hparam("pos", "timing")  # timing, none
    hparams.add_hparam("nbr_decoder_problems", 1)
    hparams.add_hparam("proximity_bias", False)
    hparams.add_hparam("causal_decoder_self_attention", True)
    hparams.add_hparam("use_pad_remover", True)
    hparams.add_hparam("self_attention_type", "dot_product")
    hparams.add_hparam("conv_first_kernel", 3)
    hparams.add_hparam("attention_variables_3d", False)
    hparams.add_hparam("use_target_space_embedding", True)
    # These parameters are only used when ffn_layer=="local_moe_tpu"
    hparams.add_hparam("moe_overhead_train", 1.0)
    hparams.add_hparam("moe_overhead_eval", 2.0)
    hparams.moe_num_experts = 16
    hparams.moe_loss_coef = 1e-3
    # If specified, use this value instead of problem name in metrics.py.
    # This is useful for programs that can automatically compare experiments side
    #   by side based on the same metric names.
    hparams.add_hparam("overload_eval_metric_name", "")
    # For making a transformer encoder unidirectional by using masked
    # attention.
    hparams.add_hparam("unidirectional_encoder", False)
    # For hard attention.
    hparams.add_hparam("hard_attention_k", 0)
    hparams.add_hparam("gumbel_noise_weight", 0.0)

    hparams.layer_preprocess_sequence = "n"
    hparams.layer_postprocess_sequence = "da"
    hparams.layer_prepostprocess_dropout = 0.1

    hparams.add_hparam('unconditional', False)
    return hparams
