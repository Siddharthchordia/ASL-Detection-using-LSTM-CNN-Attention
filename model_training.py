
from ASL import *
from keras._tf_keras.keras.saving import save_model

# create_bins()
# data_collection2()
DATA_PATH = "data2_augmented"
sequences,labels=[],[]
no_aug_seq = 400

def sequencing():
    for action in actions:
        for sequence in range(no_aug_seq):
            video=[]
            for frame_num in range(seq_length):
                res=np.load(os.path.join(DATA_PATH,action,str(sequence),f"{frame_num}.npy"))
                video.append(res)
            sequences.append(video)
            labels.append(label_map[action])
    print(np.array(sequences).shape)
    print(np.array(labels).shape)
sequencing()
X=np.array(sequences)
y=to_categorical(labels).astype(int)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
log_dir=os.path.join("Logs")
tb_callback=TensorBoard(log_dir=log_dir)

try:
    model.fit(x_train, y_train, epochs=6000,batch_size=256,verbose=2, callbacks=tb_callback)
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")


save_model(model, "models/model6.keras")
print("Model saved to model6.keras")


