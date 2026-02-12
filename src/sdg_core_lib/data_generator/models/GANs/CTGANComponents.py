import tensorflow as tf
from keras import ops
from keras.api import layers
import numpy as np
import keras


class CTGANCritic(keras.Model):
    def __init__(
        self, pac_size: int = 10, hidden: int = 256, dropout: float = 0.2, **kwargs
    ):
        super(CTGANCritic, self).__init__(**kwargs)
        self.pac_size = pac_size
        self.fc1 = layers.Dense(hidden)
        self.fc2 = layers.Dense(hidden)
        self.out = layers.Dense(1)
        self.leaky = layers.LeakyReLU(negative_slope=0.2)
        self.drop = layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pac_size": self.pac_size,
                "hidden": self.fc1.units,
                "dropout": self.drop.rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Filter out only the parameters our constructor expects
        constructor_params = {
            "pac_size": config.get("pac_size", 10),
            "hidden": config.get("hidden", 256),
            "dropout": config.get("dropout", 0.2),
        }
        return cls(**constructor_params)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        feature_dim = tf.shape(x)[1]
        remainder = batch_size % self.pac_size

        def pad_batch():
            padding_size = self.pac_size - remainder
            last_sample = tf.expand_dims(x[-1], axis=0)
            padding = tf.tile(last_sample, [padding_size, 1])
            return tf.concat([x, padding], axis=0), padding_size

        def no_padding():
            return x, 0

        x_padded, pad_size = tf.cond(remainder > 0, pad_batch, no_padding)

        x_reshaped = tf.reshape(x_padded, [-1, self.pac_size * feature_dim])

        h = self.fc1(x_reshaped)
        h = self.leaky(h)
        h = self.drop(h, training=training)
        h = self.fc2(h)
        h = self.leaky(h)
        h = self.drop(h, training=training)
        score = tf.squeeze(self.out(h), axis=1)

        def remove_padding():
            valid_groups = (batch_size + self.pac_size - 1) // self.pac_size
            return score[:valid_groups]

        def keep_all():
            return score

        final_score = tf.cond(remainder > 0, remove_padding, keep_all)

        return tf.cast(final_score, tf.float64)


def gumbel_softmax(logits, tau=0.2, hard=True):
    u = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel = -tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)
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
        self.fc1 = layers.Dense(hidden)
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(hidden)
        self.bn2 = layers.BatchNormalization()
        self.alpha_heads = [layers.Dense(1) for _ in self.modes_cont]
        self.beta_heads = [layers.Dense(m) for m in self.modes_cont]
        self.d_heads = [layers.Dense(d) for d in self.cats_disc]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "skeleton": self.skeleton,
                "modes_per_continuous_column": self.modes_cont,
                "categories_per_discrete_column": self.cats_disc,
                "hidden": self.fc1.units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Filter out only the parameters our constructor expects
        constructor_params = {
            "skeleton": config.get("skeleton"),
            "modes_per_continuous_column": config.get("modes_per_continuous_column"),
            "categories_per_discrete_column": config.get(
                "categories_per_discrete_column"
            ),
            "hidden": config.get("hidden", 256),
        }
        return cls(**constructor_params)

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

        parts = []
        c_idx, d_idx = 0, 0
        for col in self.skeleton:
            if col["feature_type"] == "continuous":
                parts.append(alphas[c_idx])
                parts.append(betas[c_idx])
                c_idx += 1
            else:
                parts.append(ds[d_idx])
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
        self.probability_mass_function_list = None
        self.row_dim = (
            sum(generator.modes_cont)
            + sum(generator.cats_disc)
            + len(generator.modes_cont)
        )

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.critic_loss_tracker]

    @tf.function
    def generate_batch_cond(self, batch_size):
        num_cats = len(self.generator.cats_disc)
        total_cond_dim = sum(self.generator.cats_disc)
        cats_disc = tf.convert_to_tensor(self.generator.cats_disc, dtype=tf.int32)

        col_indices = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=num_cats, dtype=tf.int32
        )

        relevant_pmfs = tf.gather(self.probability_mass_function_list, col_indices)

        cat_indices = tf.random.categorical(tf.math.log(relevant_pmfs), num_samples=1)
        cat_indices = tf.cast(tf.squeeze(cat_indices, axis=1), tf.int32)

        offsets_table = tf.concat([[0], tf.cumsum(cats_disc)[:-1]], axis=0)
        batch_offsets = tf.gather(offsets_table, col_indices)

        global_hot_indices = batch_offsets + cat_indices
        row_indices = tf.range(batch_size)
        scatter_indices = tf.stack([row_indices, global_hot_indices], axis=1)

        cond_batch = tf.scatter_nd(
            indices=scatter_indices,
            updates=tf.ones([batch_size], dtype=tf.float32),
            shape=[batch_size, total_cond_dim],
        )

        return cond_batch

    @staticmethod
    @tf.function
    def sample_real_data(train_tensor, cond, discrete_onehot_indexes):
        if tf.rank(cond) == 1:
            cond = tf.expand_dims(cond, axis=0)

        discrete_indices = tf.constant(discrete_onehot_indexes, dtype=tf.int32)
        cond_indices = tf.cast(tf.argmax(cond, axis=1), tf.int32)
        target_columns = tf.gather(discrete_indices, cond_indices)

        def sample_single_row(col):
            mask = tf.equal(train_tensor[:, col], 1.0)
            elems = tf.boolean_mask(train_tensor, mask)
            num_elems = tf.shape(elems)[0]
            tf.Assert(num_elems > 0, ["No row found for condition!"])
            logits = tf.zeros([1, num_elems])
            random_idx = tf.random.categorical(logits, 1)
            random_idx = tf.cast(tf.reshape(random_idx, []), tf.int32)
            return tf.gather(elems, random_idx)

        return tf.map_fn(
            sample_single_row, target_columns, fn_output_signature=train_tensor.dtype
        )

    @tf.function
    def train_critic(self, real_data, z, cond):
        with tf.GradientTape() as tape:
            fake_data, _, _, _ = self.generator([z, cond], training=True)

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
            cond_loss = -tf.reduce_mean(
                tf.reduce_sum(cond * tf.math.log(d_logits + 1e-8), axis=1)
            )
            g_loss = adv_loss + tf.cast(cond_loss, tf.float64)

        grads_g = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads_g, self.generator.trainable_variables)
        )
        return g_loss

    def get_pmfs(self, train_data):
        onehot_all = tf.gather(train_data, self.onehot_discrete_indexes, axis=1)
        pmfs = []
        curr = 0
        for sz in self.generator.cats_disc:
            chunk = onehot_all[:, curr : curr + sz]
            log_freqs = tf.math.log(tf.reduce_sum(chunk, axis=0) + 1.0)
            pmfs.append(log_freqs / tf.reduce_sum(log_freqs))
            curr += sz
        return pmfs

    def train_step(self, data):
        batch = ops.shape(data)[0]
        self.row_dim = ops.shape(data)[1]
        z = tf.random.normal([batch, self.row_dim - sum(self.generator.cats_disc)])
        cond = self.generate_batch_cond(batch)
        real_batch = CTGANModel.sample_real_data(
            self._train_data, cond, self.onehot_discrete_indexes
        )
        c_loss = self.train_critic(real_batch, z, cond)
        g_loss = self.train_gen(z, cond)

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
        if self.generator is None or self.probability_mass_function_list is None:
            raise RuntimeError(
                "In order to generate some data you need to fit a dataset first!"
            )

        z = keras.random.normal(
            shape=(batch_size, self.row_dim - sum(self.generator.cats_disc)), seed=42
        )
        cond = self.generate_batch_cond(batch_size)
        gen_x, _, _, _ = self.generator([z, cond], training=False)
        return ops.convert_to_numpy(gen_x)
