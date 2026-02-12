import tensorflow as tf
from keras import ops
from keras.api import layers
import numpy as np
import keras


class CTGANCritic(keras.Model):
    """
    The critic follows the PacGAN framework, evaluating the realness
    of multiple samples together to penalize mode collapse.
    """

    def __init__(
        self, pac_size: int = 10, hidden: int = 256, dropout: float = 0.2, **kwargs
    ):
        super(CTGANCritic, self).__init__(**kwargs)

        self.pac_size = pac_size

        # Layers
        self.fc1 = layers.Dense(hidden)
        self.fc2 = layers.Dense(hidden)
        self.out = layers.Dense(1)

        # Activations and Regularization
        self.leaky = layers.LeakyReLU(negative_slope=0.2)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        """
        x: (B, row_dim + cond_dim)
        Note: The concatenation of row and cond should happen before
        passing x to this call, or inside it if preferred.
        """
        from loguru import logger
        logger.info(tf.shape(x))
        # Get dynamic batch size
        batch_size = tf.shape(x)[0]

        # Logic check for PacGAN grouping
        # In Keras, we use tf.Assert for runtime checks inside the graph
        tf.debugging.assert_equal(
            batch_size % self.pac_size,
            0,
            message=f"Batch size must be divisible by pac_size {self.pac_size}",
        )

        # Reshape to (B // pac_size, pac_size * feature_dim)
        # -1 allows TensorFlow to infer the kp (batch // pac_size) dimension
        x = tf.reshape(x, [-1, self.pac_size * tf.shape(x)[1]])

        # Forward Pass
        h = self.fc1(x)
        h = self.leaky(h)
        h = self.drop(h, training=training)

        h = self.fc2(h)
        h = self.leaky(h)
        h = self.drop(h, training=training)

        # Output score: (kp, 1) -> (kp,)
        score = tf.squeeze(self.out(h), axis=1)

        return tf.cast(score, tf.float64)


def gumbel_softmax(logits, tau=0.2, hard=True):
    """Differentiable sampling for discrete columns."""
    U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel = -tf.math.log(-tf.math.log(U + 1e-20) + 1e-20)
    y = tf.nn.softmax((logits + gumbel) / tau)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


class CTGANGenerator(keras.Model):
    def __init__(
        self,
        skeleton,
        modes_per_continuous_column,
        categories_per_discrete_column,
        hidden=256,
    ):
        super().__init__()
        self.skeleton = skeleton
        self.tau = 0.2
        self.modes_cont = modes_per_continuous_column
        self.cats_disc = categories_per_discrete_column

        # Shared architecture
        self.fc1 = layers.Dense(hidden)
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(hidden)
        self.bn2 = layers.BatchNormalization()

        # Output heads
        self.alpha_heads = [layers.Dense(1) for _ in self.modes_cont]
        self.beta_heads = [layers.Dense(m) for m in self.modes_cont]
        self.d_heads = [layers.Dense(d) for d in self.cats_disc]

    def call(self, inputs, training=False):
        z, cond = inputs
        h = tf.concat([z, cond], axis=1)
        h = tf.nn.relu(self.bn1(self.fc1(h), training=training))
        h = tf.nn.relu(self.bn2(self.fc2(h), training=training))

        alphas, betas, ds = [], [], []
        for i in range(len(self.alpha_heads)):
            alphas.append(tf.nn.tanh(self.alpha_heads[i](h)))
            betas.append(gumbel_softmax(self.beta_heads[i](h), self.tau))
        for j in range(len(self.d_heads)):
            ds.append(gumbel_softmax(self.d_heads[j](h), self.tau))

        # INTERLEAVED RECONSTRUCTION
        parts = []
        c_idx, d_idx = 0, 0
        for col in self.skeleton:
            if col["feature_type"] == "continuous":
                parts.append(alphas[c_idx])  # The scalar value
                parts.append(betas[c_idx])  # The mode one-hot
                c_idx += 1
            else:
                parts.append(ds[d_idx])  # The categorical one-hot
                d_idx += 1

        full_row = tf.concat(parts, axis=1)
        return full_row, alphas, betas, ds


class CTGANModel(keras.Model):
    def __init__(
        self,
        generator: CTGANGenerator,
        critic: CTGANCritic,
        onehot_discrete_indexes: list[int] | None = None,
    ):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.onehot_discrete_indexes = onehot_discrete_indexes
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.critic_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self._train_data = None

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.critic_loss_tracker]

    @tf.function
    def generate_batch_cond(self, batch_size):
        num_cats = len(self.generator.cats_disc)
        total_cond_dim = sum(self.generator.cats_disc)
        cats_disc = tf.convert_to_tensor(self.generator.cats_disc, dtype=tf.int32)

        # 1. Pick a random discrete column for EVERY item in the batch
        # Shape: (batch_size,)
        col_indices = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=num_cats, dtype=tf.int32
        )

        # 2. Pick a category for each chosen column
        # We use a loop-free trick: gather all PMFs and select based on col_indices
        # Shape: (batch_size, max_categories_per_col)
        relevant_pmfs = tf.gather(self.pmfs, col_indices)

        # Sample from each row's specific PMF
        # tf.random.categorical handles batches automatically!
        cat_indices = tf.random.categorical(tf.math.log(relevant_pmfs), num_samples=1)
        cat_indices = tf.cast(tf.squeeze(cat_indices, axis=1), tf.int32)

        # 3. Calculate global offsets for each condition
        # Using cumulative sum to find the starting position of each discrete block
        offsets_table = tf.concat([[0], tf.cumsum(cats_disc)[:-1]], axis=0)
        batch_offsets = tf.gather(offsets_table, col_indices)

        # 4. Create the global one-hot condition vectors
        # We calculate the specific index in the flat vector for each batch item
        global_hot_indices = batch_offsets + cat_indices

        # Create a batch of row indices [0, 1, 2, ..., N-1]
        row_indices = tf.range(batch_size)

        # Pack row and column coordinates: [[0, idx0], [1, idx1], ...]
        scatter_indices = tf.stack([row_indices, global_hot_indices], axis=1)

        # Update a zero matrix of shape (batch_size, total_cond_dim)
        cond_batch = tf.scatter_nd(
            indices=scatter_indices,
            updates=tf.ones([batch_size], dtype=tf.float32),
            shape=[batch_size, total_cond_dim],
        )

        return cond_batch

    @staticmethod
    @tf.function
    def sample_real_data(train_tensor, cond, discrete_onehot_indexes):
        # 1. Force cond to be 2D if it's 1D (Batching)
        if tf.rank(cond) == 1:
            cond = tf.expand_dims(cond, axis=0)

        discrete_indices = tf.constant(discrete_onehot_indexes, dtype=tf.int32)

        # 2. Get the column indices for the whole batch
        cond_indices = tf.cast(tf.argmax(cond, axis=1), tf.int32)

        # 3. Gather the target columns from your index list
        target_columns = tf.gather(discrete_indices, cond_indices)

        def sample_single_row(col):
            # col will be a scalar here because map_fn unstacks the vector
            # To use it in boolean_mask, we just use it directly
            mask = tf.equal(train_tensor[:, col], 1.0)
            elems = tf.boolean_mask(train_tensor, mask)

            num_elems = tf.shape(elems)[0]
            tf.Assert(num_elems > 0, ["No row found for condition!"])

            # Uniform sampling
            logits = tf.zeros([1, num_elems])
            random_idx = tf.random.categorical(logits, 1)
            random_idx = tf.cast(tf.reshape(random_idx, []), tf.int32)

            return tf.gather(elems, random_idx)

        # 4. map_fn works on target_columns now because it's at least shape (1,)
        return tf.map_fn(
            sample_single_row, target_columns, fn_output_signature=train_tensor.dtype
        )

    @tf.function
    def train_critic(self, real_data, z, cond):
        with tf.GradientTape() as tape:
            fake_data, _, _, _ = self.generator([z, cond], training=True)

            # Reshape for PacGAN
            # Note: real_data and fake_data are (Batch, DataDim)
            # Critic expects (Batch/Pac, Pac * DataDim)

            # WGAN-GP Gradient Penalty
            alpha = tf.random.uniform([ops.shape(real_data)[0], 1], 0.0, 1.0)
            alpha = tf.cast(alpha, tf.float64)
            real_data = tf.cast(real_data, tf.float64)
            fake_data = tf.cast(fake_data, tf.float64)
            interpolated = alpha * real_data + (1 - alpha) * fake_data
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.critic(interpolated, training=True)
            grads = gp_tape.gradient(pred, [interpolated])[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
            gp = tf.cast(tf.reduce_mean((norm - 1.0) ** 2) * 10.0, tf.float64)
            real_score = self.critic(real_data, training=True)
            fake_score = self.critic(fake_data, training=True)
            c_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp

        grads_c = tape.gradient(c_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(grads_c, self.critic.trainable_variables)
        )
        return c_loss

    @tf.function
    def train_gen(self, z, cond):
        with tf.GradientTape() as tape:
            fake_data, _, _, d_list = self.generator([z, cond], training=True)
            fake_score = self.critic(fake_data, training=True)

            adv_loss = -tf.reduce_mean(fake_score)
            d_logits = tf.concat(d_list, axis=1)
            # Log-frequency weighted cross-entropy
            cond_loss = -tf.reduce_mean(
                tf.reduce_sum(cond * tf.math.log(d_logits + 1e-8), axis=1)
            )
            g_loss = adv_loss + tf.cast(cond_loss, tf.float64)

        grads_g = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads_g, self.generator.trainable_variables)
        )
        return g_loss

    def _get_pmfs(self, train_data):
        """
        Plucks intertwined columns using self.disc_idxs to calculate log-frequencies.
        """
        # Pluck only the one-hot columns from the intertwined dataset
        onehot_all = tf.gather(train_data, self.onehot_discrete_indexes, axis=1)
        pmfs = []
        curr = 0
        for sz in self.generator.cats_disc:
            # Slice the one-hot block for THIS specific discrete feature
            chunk = onehot_all[:, curr : curr + sz]
            log_freqs = tf.math.log(tf.reduce_sum(chunk, axis=0) + 1.0)
            pmfs.append(log_freqs / tf.reduce_sum(log_freqs))
            curr += sz
        return pmfs

    def train_step(self, data):
        # TODO: Note that this is generating a cond vector which is the same for all batch
        # TODO: This needs to be fixed
        # TODO: The z vector needs to be properly created since it represent
        # the noise simulating continous values (this should be tested tho)
        print(data)
        self.pmfs = tf.convert_to_tensor(
            self._get_pmfs(self._train_data), dtype=tf.float64
        )
        batch = ops.shape(data)[0]
        self.row_dim = ops.shape(data)[1]
        z = tf.random.normal([batch, self.row_dim - sum(self.generator.cats_disc)])
        cond = self.generate_batch_cond(batch)
        real_batch = CTGANModel.sample_real_data(
            self._train_data, cond, self.onehot_discrete_indexes
        )
        c_loss = self.train_critic(real_batch, z, cond)
        g_loss = self.train_gen(z, cond)
        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.critic_loss_tracker.update_state(c_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.critic_loss_tracker.result(),
        }

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.generator.compile(g_optimizer)
        self.critic.compile(d_optimizer)

    def generate(self, batch_size: int = 100) -> np.ndarray:
        """
        Generates synthetic data samples.
        """
        if self.generator is None or self.pmfs is None:
            raise RuntimeError(
                "In order to generate some data you need to fit a dataset first!"
            )

        # 1. Sample noise vector z using Keras ops
        z = keras.random.normal(shape=(batch_size, self.row_dim), seed=42)

        # 2. Sample conditional vector
        # Note: self.sample_cond already handles batch_size and internal state
        cond = self.generate_batch_cond(batch_size)

        # 3. Pass through generator
        # Keras models use the 'training=False' flag for inference
        gen_x, _, _, _ = self.generator([z, cond], training=False)

        # 4. Convert to numpy
        # In Keras 3, we use ops.convert_to_numpy() to safely move data from GPU/TPU to CPU
        return ops.convert_to_numpy(gen_x)
