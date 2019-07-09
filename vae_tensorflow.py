import tensorflow as tf
from dataPrep.data_generator import DataGenerator
from config.readconfig import Config
from datetime import datetime
from tqdm import tqdm
import os
from numpy import squeeze
from matplotlib import pyplot as plt



class Model():

    def __init__(self):
        print("Building Model...")

    def __new__(self,config,inputs):
        outputs,inputs,var,mean = self.create_architecture(self,config,inputs)
        return outputs,inputs,var,mean

    def create_architecture(self,config,inputs):


        with tf.name_scope('encoder'):
            x = tf.keras.layers.Dense(784,name='Dense_0',activation='relu')(inputs)
            x = tf.keras.layers.Dense(512,name='Dense_1',activation='relu')(x)
            x = tf.keras.layers.Dense(256,name='Dense_2',activation='relu')(x)
            x = tf.keras.layers.Dense(128,name='Dense_3',activation='relu')(x)
            x = tf.keras.layers.Dense(64,name='Dense_4',activation='relu')(x)

            input_image = tf.reshape(inputs,(64,28,28,1))
            tf.summary.image('input',tensor=input_image,max_outputs=1)

        with tf.name_scope('Z_mean_var'):
            mean = tf.keras.layers.Dense(config.latent_size,name='Zmean')(x)
            variance = tf.keras.layers.Dense(config.latent_size,name='Z_log_var')(x)
            mean = tf.cast(mean,tf.float64)
            variance = tf.cast(variance,tf.float64)

        def sampling(args):
            with tf.name_scope('sampling'):

                z_mean , z_log_var = args

                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random_normal(shape=(batch,dim),dtype=tf.float64)
                sampled_latent_vector = z_mean + tf.exp(0.5 * z_log_var) * epsilon
                return sampled_latent_vector

        with tf.name_scope('Z'):
            z = tf.keras.layers.Lambda(sampling,output_shape=(10,),name='z')([mean,variance])


        with tf.name_scope('decoder'):
            x = tf.keras.layers.Dense(64,activation='relu',name='Dense_0')(z)
            x = tf.keras.layers.Dense(128,activation='relu',name='Dense_1')(x)
            x = tf.keras.layers.Dense(256,activation='relu',name='Dense_2')(x)
            x = tf.keras.layers.Dense(512,activation='relu',name='Dense_3')(x)
            outputs = tf.keras.layers.Dense(784,activation='sigmoid',name='Dense_Out')(x)
            images = tf.reshape(outputs,shape=(64,28,28,1),name='image')
            tf.summary.image(name='output',tensor=images,max_outputs=1)

        return outputs,inputs,variance,mean


def compute_cost(config,outputs,inputs,variance,mean):

    with tf.name_scope('Cost'):

        # Reconstruction loss
        encode_decode_loss = inputs*tf.log(1e-10 + outputs) + (1-inputs) * tf.log(1e-10 + 1 - outputs)
        encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
        # KL Divergence loss
        kl_div_loss = 1 + variance - tf.square(mean) - tf.exp(variance)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        vae_loss = tf.reduce_mean(encode_decode_loss + kl_div_loss,name='vloss')
        tf.summary.scalar(name='vae_loss',tensor=vae_loss)

        return vae_loss


def optimize(config,loss):

    with tf.name_scope('optimize'):
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
    return optimizer

'''Intiliaze Model Params'''

config = Config('./config/model_config.yaml')
data = DataGenerator(config)
train = data.dataset
val  = data.dev_dataset

'''Build Model'''
handle = tf.placeholder(tf.string,shape=[],name='input_handle')
iterator = tf.data.Iterator.from_string_handle(handle,output_shapes=train.output_shapes,output_types=train.output_types)
inputs = iterator.get_next()[0]

outputs,inputs,variance,mean = Model(config,inputs)
cost = compute_cost(config,outputs,inputs,variance,mean)
optimizer = optimize(config,cost)

'''Session Initializers'''

training_iterator = train.make_one_shot_iterator()
validation_iterator = val.make_one_shot_iterator()

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
saver = tf.train.Saver()
logpath = config.logs+'train'+'/'+str(datetime.now())
train_writer = tf.summary.FileWriter(logdir=logpath)
logpath = config.logs+'validation'+'/'+str(datetime.now())
val_writer = tf.summary.FileWriter(logdir=logpath)


'''Session Train Run'''

if config.type == 'train':

    with tf.control_dependencies([outputs,cost,optimizer]):
        with tf.Session() as sess:
            sess.run(init)
            training_handler,validation_handler = sess.run([training_iterator.string_handle(),validation_iterator.string_handle()])

            for each in range(config.no_epochs):

                batch_bar = tqdm(iterable=range(config.no_steps),desc="Epoch-{}".format(each))
                for batch in batch_bar:
                    _,t_loss,summary = sess.run([optimizer,cost,merged],feed_dict={handle:training_handler})
                    train_writer.add_summary(summary,batch)


                    v_loss,summary = sess.run([cost,merged],feed_dict={handle:validation_handler})
                    val_writer.add_summary(summary,batch)
                    batch_bar.set_description("Train Loss-->{}, Val Loss-->{}".format(str(t_loss)[:5],str(v_loss)[:5]))

                val_writer.flush()
                train_writer.flush()
                saver.save(sess,config.modeldump)
                print("Model Saved at {}".format(config.modeldump))

else:
    with tf.Session() as sess:
        print("Restoring..")
        saver = tf.train.import_meta_graph(config.modeldump + '.meta')
        saver.restore(sess, config.modeldump)

        graph = tf.get_default_graph()
        # nodes = [n.name for n in tf.get_default_graph().as_graph_def().node if n.name.startswith('decoder/im')]
        # print(nodes)
        loss = graph.get_tensor_by_name('Cost/vloss:0')
        outputs = graph.get_tensor_by_name('decoder/image:0')
        training_handler,validation_handler = sess.run([training_iterator.string_handle(),validation_iterator.string_handle()])
        test_outputs = []

        for _ in range(2):
            tout = sess.run([outputs],feed_dict={handle:training_handler})
            test_outputs.append(tout)

        fig=plt.figure(figsize=(64,64))
        columns = 8
        rows = 8
        for each_batch in test_outputs[0]:
            for i,img in enumerate(each_batch):
                img = squeeze(img)
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(img)
            plt.show()
