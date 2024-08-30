# imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as image
from scipy.spatial.distance import euclidean
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from scipy.stats import sem


# utilities:
import os
from PIL import Image

# Keras:
from keras.models import Model

# VGG:
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# clip:
import clip
import torch

# torch:
import torch
import torchvision.models as models
from feature_net import VGG16FeatureNet

# Global Variable:
CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
BASE_FOLDER = os.path.dirname(CODE_FOLDER)

class CompositeEffectExp:
    """
    Run a composite effect experiment on a deep learning algorithm trained on face recognition.
    Initialize -> run_experiment -> analyze.
    """
    
    ### Initialization ###

    def __init__(self, folder=BASE_FOLDER, model='VGG', exp_type='both', random_state=42):
        self.path = folder
        self.results_path = os.path.join(BASE_FOLDER, 'Results')
        self.exp_type = exp_type
        self.upright_aligned_faces_path = os.path.join(BASE_FOLDER, 'Stimuli', 'Aligned')
        self.upright_misaligned_faces_path = os.path.join(BASE_FOLDER, 'Stimuli', 'Misaligned')
        self.inverted_aligned_faces_path = os.path.join(BASE_FOLDER, 'Inverted_Stimuli', 'Aligned')
        self.inverted_misaligned_faces_path = os.path.join(BASE_FOLDER, 'Inverted_Stimuli', 'Misaligned')                                              
        self.code_path = os.path.join(folder, 'Code')
        self.random_state = random_state  # random state for reproducibility
        self.model_type = model
        self.upright_aligned_features = {}
        self.upright_misaligned_features = {}
        self.inverted_aligned_features = {}
        self.inverted_misaligned_features = {}
        self.df_all = None  # replaced with pd.DataFrame after run_experiment() command
        self.df = None      # replaced with pd.DataFrame after analyze() command
        self.results = None # replaced with scipy ttest_results after analyze() comand
        
        # Initialize model hash
        self.model_hash = {
            'VGG': self.get_vgg,
            'resnet': self.get_resnet,
            'senet': self.get_senet,
            'clip': self.get_clip,
            'FeatureNet': self.get_featurenet
        }
        
        # Set the model using the model hash table
        if self.model_type in self.model_hash:
            self.model = self.model_hash[model]()
        else:
            raise ValueError(f"Model '{model}' not recognized. Available models: {list(self.model_hash.keys())}")

    def get_vgg(self):
        vgg = VGGFace()
        return Model(inputs=vgg.input, outputs=vgg.get_layer('fc7').output)
    
    def get_resnet(self):
        resnet = VGGFace(model='resnet50')
        return Model(inputs=resnet.input, outputs=resnet.get_layer(resnet.layers[-2].name).output)
    
    def get_senet(self):
        senet = VGGFace(model='senet50')
        return Model(inputs=senet.input, outputs=senet.get_layer(senet.layers[-2].name).output)

    def get_clip(self):
        pass

    def get_featurenet(self):
        model = torch.load(os.path.join(CODE_FOLDER, 'FeatureNet.pt'))
        layer_index = 3  
        feature_extractor = VGG16FeatureNet(model, layer_index)
        feature_extractor.eval()
        return feature_extractor


    ### Experiment ### 
    
    def preprocess_image(self, img_path, new_size=(224, 224)):
        img = Image.open(img_path).convert('RGB')          # Open and convert image to RGB
        resized_img = img.resize(new_size)                 # Resize image
        img_array = np.array(resized_img, dtype='float32') # Convert image to numpy array with dtype float32
        img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
        img_array = preprocess_input(img_array)            # Preprocess the input - apply normalization according to imagenet standards
        return img_array

    def extract_features(self, img_path):
        if self.model_type == 'clip':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            img = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                return np.array(clip_model.encode_image(img).ravel())
        
        elif self.model_type == 'FeatureNet':
            # Ensure the tensor is on the correct device (CPU or GPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)

            # Load the image and preprocess it
            img_data = self.preprocess_image(img_path).copy()
            tensor_input = torch.tensor(img_data).permute(0, 3, 1, 2)

            # Extract features
            with torch.no_grad():
                features = self.model(tensor_input)

            # Convert the features tensor to a numpy array if needed
            features_array = features.cpu().numpy()

            # Flatten the array and return it
            return features_array.flatten()

        else: # VGG / RESNET / SENET    
            img_data = self.preprocess_image(img_path)
            features = self.model.predict(img_data)
            return features.flatten()

    def fill_features_hash_tables(self):
        if self.exp_type == 'both' or self.exp_type == 'upright':
            # Iterate over upright aligned faces and extract features
            for img_name in [f for f in os.listdir(self.upright_aligned_faces_path) if f.endswith(('bmp', 'png', 'jpg', 'jpeg'))]:
                img_path = os.path.join(self.upright_aligned_faces_path, img_name)
                if os.path.isfile(img_path):
                    self.upright_aligned_features[img_name] = self.extract_features(img_path)

            # Iterate over upright misaligned faces and extract features
            for img_name in [f for f in os.listdir(self.upright_misaligned_faces_path) if f.endswith(('bmp', 'png', 'jpg', 'jpeg'))]:
                img_path = os.path.join(self.upright_misaligned_faces_path, img_name)
                if os.path.isfile(img_path):
                    self.upright_misaligned_features[img_name] = self.extract_features(img_path)

        if self.exp_type == 'both' or self.exp_type == 'inverted':
            # Iterate over inverted aligned faces and extract features
            for img_name in [f for f in os.listdir(self.inverted_aligned_faces_path) if f.endswith(('bmp', 'png', 'jpg', 'jpeg'))]:
                img_path = os.path.join(self.inverted_aligned_faces_path, img_name)
                if os.path.isfile(img_path):
                    self.inverted_aligned_features[img_name] = self.extract_features(img_path)

            # Iterate over inverted misaligned faces and extract features
            for img_name in [f for f in os.listdir(self.inverted_misaligned_faces_path) if f.endswith(('bmp', 'png', 'jpg', 'jpeg'))]:
                img_path = os.path.join(self.inverted_misaligned_faces_path, img_name)
                if os.path.isfile(img_path):
                    self.inverted_misaligned_features[img_name] = self.extract_features(img_path)




    def get_congruency(self, id1_A, id2_A, id1_B, id2_B):
        if (id1_A == id1_B and id2_A == id2_B) or (id1_A != id1_B and id2_A != id2_B):
            return 'congruent'
        else:
            return 'incongruent'

    def run_experiment(self):
        self.fill_features_hash_tables()

        data = []
        upright_aligned_keys = list(self.upright_aligned_features.keys())
        upright_misaligned_keys = list(self.upright_misaligned_features.keys())
        inverted_aligned_keys = list(self.inverted_aligned_features.keys())
        inverted_misaligned_keys = list(self.inverted_misaligned_features.keys())

        if self.exp_type == 'both' or self.exp_type == 'upright':
            # Process   upright aligned features
            for i in range(len(upright_aligned_keys)):
                img1 = upright_aligned_keys[i]
                features1 = self.upright_aligned_features[img1]
                id1_1, id1_2 = img1.split('_')[1:3]
                for j in range(i + 1, len(upright_aligned_keys)):
                    img2 = upright_aligned_keys[j]
                    features2 = self.upright_aligned_features[img2]
                    id2_1, id2_2 = img2.split('_')[1:3]
                    identity = 'same' if id1_1 == id2_1 else 'different'
                    congruency = self.get_congruency(id1_1, id1_2, id2_1, id2_2)
                    distance = euclidean(features1, features2)
                    data.append([img1, img2,'upright', 'aligned', identity, congruency, distance])

            # Process misaligned features
            for i in range(len(upright_misaligned_keys)):
                img1 = upright_misaligned_keys[i]
                features1 = self.upright_misaligned_features[img1]
                id1_1, id1_2 = img1.split('_')[1:3]
                for j in range(i + 1, len(upright_misaligned_keys)):
                    img2 = upright_misaligned_keys[j]
                    features2 = self.upright_misaligned_features[img2]
                    id2_1, id2_2 = img2.split('_')[1:3]
                    identity = 'same' if id1_1 == id2_1 else 'different'
                    congruency = self.get_congruency(id1_1, id1_2, id2_1, id2_2)
                    distance = euclidean(features1, features2)
                    data.append([img1, img2, 'upright', 'misaligned', identity, congruency, distance])

        if self.exp_type == 'both' or self.exp_type == 'inverted':
            # Process inverted aligned features
            for i in range(len(inverted_aligned_keys)):
                img1 = inverted_aligned_keys[i]
                features1 = self.inverted_aligned_features[img1]
                id1_1, id1_2 = img1.split('_')[1:3]
                for j in range(i + 1, len(inverted_aligned_keys)):
                    img2 = inverted_aligned_keys[j]
                    features2 = self.inverted_aligned_features[img2]
                    id2_1, id2_2 = img2.split('_')[1:3]
                    identity = 'same' if id1_1 == id2_1 else 'different'
                    congruency = self.get_congruency(id1_1, id1_2, id2_1, id2_2)
                    distance = euclidean(features1, features2)
                    data.append([img1, img2,'inverted', 'aligned', identity, congruency, distance])

            # Process misaligned features
            for i in range(len(inverted_misaligned_keys)):
                img1 = inverted_misaligned_keys[i]
                features1 = self.inverted_misaligned_features[img1]
                id1_1, id1_2 = img1.split('_')[1:3]
                for j in range(i + 1, len(inverted_misaligned_keys)):
                    img2 = inverted_misaligned_keys[j]
                    features2 = self.inverted_misaligned_features[img2]
                    id2_1, id2_2 = img2.split('_')[1:3]
                    identity = 'same' if id1_1 == id2_1 else 'different'
                    congruency = self.get_congruency(id1_1, id1_2, id2_1, id2_2)
                    distance = euclidean(features1, features2)
                    data.append([img1, img2, 'inverted', 'misaligned', identity, congruency, distance])

        # Save the full DataFrame:
        df = pd.DataFrame(data, columns=['image1', 'image2','condition', 'alignment', 'identity', 'congruency', 'distance'])
        self.df_all = df        

    def partial_design(self):
        # Filter the DataFrame
        df = self.df_all[
            ((self.df_all['congruency'] == 'congruent') & (self.df_all['identity'] == 'different')) |
            ((self.df_all['congruency'] == 'incongruent') & (self.df_all['identity'] == 'same'))]
        

        if self.exp_type != 'both': # either upright or inverted
            # sample the same number of same and different pairs:
            same_df = df[df['identity'] == 'same']
            different_df = df[df['identity'] == 'different']
            n_sample = len(same_df) // 2
            sampled_diff_df = pd.concat([
                            different_df[different_df['alignment'] == 'aligned'].sample(n=n_sample, random_state=self.random_state),
                            different_df[different_df['alignment'] == 'misaligned'].sample(n=n_sample, random_state=self.random_state)
                            ]).reset_index(drop=True)
            
        else: # both upright and inverted
            # sample the same number of upright_same, upright_different, inverted_same, inverted_different pairs:
            same_df = df[df['identity'] == 'same']
            upright_same_df = df[(df['condition'] == 'upright') & (df['identity'] == 'same')]
            upright_diff_df = df[(df['condition'] == 'upright') & (df['identity'] == 'different')]
            inverted_same_df = df[(df['condition'] == 'inverted') & (df['identity'] == 'same')]
            inverted_diff_df = df[(df['condition'] == 'inverted') & (df['identity'] == 'different')]
            n_sample = len(upright_same_df) // 2
            sampled_diff_df = pd.concat([
                            upright_diff_df[upright_diff_df['alignment'] == 'aligned'].sample(n=n_sample, random_state=self.random_state),
                            upright_diff_df[upright_diff_df['alignment'] == 'misaligned'].sample(n=n_sample, random_state=self.random_state),
                            inverted_diff_df[inverted_diff_df['alignment'] == 'aligned'].sample(n=n_sample, random_state=self.random_state),
                            inverted_diff_df[inverted_diff_df['alignment'] == 'misaligned'].sample(n=n_sample, random_state=self.random_state)
                            ]).reset_index(drop=True)
        
        # update self.df
        self.df = pd.concat([same_df, sampled_diff_df]).reset_index(drop=True)
        filtered_df = self.df

        # Check lengths of each group
        print(f"Lengths - aligned_same: {len(filtered_df[(filtered_df['alignment'] == 'aligned') & (filtered_df['identity'] == 'same')])}, aligned_different: {len(filtered_df[(filtered_df['alignment'] == 'aligned') & (filtered_df['identity'] == 'different')])}, misaligned_same: {len(filtered_df[(filtered_df['alignment'] == 'misaligned') & (filtered_df['identity'] == 'same')])}, misaligned_different: {len(filtered_df[(filtered_df['alignment'] == 'misaligned') & (filtered_df['identity'] == 'different')])}")
        
        if len(filtered_df[(filtered_df['alignment'] == 'aligned') & (filtered_df['identity'] == 'same')]) < 2 or len(filtered_df[(filtered_df['alignment'] == 'aligned') & (filtered_df['identity'] == 'different')]) < 2 or len(filtered_df[(filtered_df['alignment'] == 'misaligned') & (filtered_df['identity'] == 'same')]) < 2 or len(filtered_df[(filtered_df['alignment'] == 'misaligned') & (filtered_df['identity'] == 'different')]) < 2:
            raise ValueError("Insufficient data points in one or more conditions for statistical analysis.")
        
        if self.exp_type != 'both':
            # Run two-way ANOVA
            model = ols('distance ~ C(identity) * C(alignment)', data=filtered_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # Print ANOVA table
            print(anova_table)

            # Plot bar plot:
            means = filtered_df.groupby(['alignment', 'identity'])['distance'].mean().reset_index()
            g = sns.catplot(x='alignment', y='distance', hue='identity', data=means, kind='bar', palette='muted')
            g.despine(right=True, top=True)
            g.set_ylabels('Mean Distance')
            g.set_axis_labels('Alignment', 'Mean Distance')
            g.set_xticklabels(['Aligned', 'Misaligned'])
            model_name = self.model_type[0].upper() + self.model_type[1:]
            exp_type = 'Inverted' if self.exp_type=='inverted' else 'Upright'
            plt.title(f'{model_name} - {exp_type}', fontsize=20)
            plt.show()
            
        
        else: # aka 'both'
            # Run three-way ANOVA
            model = ols('distance ~ C(condition) * C(identity) * C(alignment)', data=filtered_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # Print ANOVA table
            print(anova_table)

            # Calculate means and SEM for error bars
            grouped = filtered_df.groupby(['condition', 'alignment', 'identity'])['distance']
            means = grouped.mean().reset_index()
            errors = grouped.apply(sem).reset_index(name='sem')

            # Merge means and errors
            means = means.merge(errors, on=['condition', 'alignment', 'identity'])

            # Plot bar plot with error bars
            g = sns.catplot(x='alignment', y='distance', hue='identity', col='condition', data=means, kind='bar', palette='muted')

            # Add error bars
            for ax in g.axes.flat:
                for i, bar in enumerate(ax.patches):
                    # Get the data for the current bar
                    condition = means.loc[i, 'condition']
                    alignment = means.loc[i, 'alignment']
                    identity = means.loc[i, 'identity']
                    error = means.loc[i, 'sem']

                    # Calculate the x-coordinate of the error bar
                    x = bar.get_x() + bar.get_width() / 2

                    # Draw the error bar
                    ax.errorbar(x, bar.get_height(), yerr=error, fmt='k', capsize=5) 
            
            
            # Manually set titles
            condition_titles = {'upright': 'Upright', 'inverted': 'Inverted'}
            for ax in g.axes.flat:
                condition = ax.get_title().split(' = ')[1]
                ax.set_title(condition_titles[condition])

            g.despine(right=True, top=True)
            g.set_ylabels('Mean Distance')
            g.set_axis_labels('Alignment', 'Mean Distance')
            g.set_xticklabels(['Aligned', 'Misaligned'])

            # Adjust the distance between the plot and the suptitle
            plt.subplots_adjust(top=0.8)

            # Set the suptitle
            model_name = self.model_type[0].upper() + self.model_type[1:]
            plt.suptitle(f'{model_name}', fontsize=20, y=1.1)

            plt.show()

            # save figure
            fig_name = f"bar_plot_{self.model_type}.png"
            fig_path = os.path.join(self.results_path, fig_name)
            g.savefig(fig_path)
        
        self.results = anova_table
        
        # save the results to a csv file
        output_file_name = f"results_{self.model_type}.csv"
        output_file_path = os.path.join(self.results_path, output_file_name)
        self.results.to_csv(output_file_path)

        return anova_table
    

    def post_hoc(self):
        if self.results is None:
            raise ValueError("Run the experiment first to get results.")
        
        # Check if the 3-way interaction is significant
        if 'C(condition):C(identity):C(alignment)' not in self.results.index:
            print("3-way interaction term not found in results.")
            return
        
        interaction_p = self.results.loc['C(condition):C(identity):C(alignment)', 'PR(>F)']
        
        if interaction_p > 0.05:
            print("3-way interaction is not significant. No post hoc analysis needed.")
            return
        
        print("3-way interaction is significant. Performing post hoc analysis.")
        
        # Prepare a list to store results
        post_hoc_results = []
        
        # Perform ANOVAs for each condition of the 'condition' variable
        conditions = self.df['condition'].unique()
        for cond in conditions:
            subset = self.df[self.df['condition'] == cond]
            model = ols('distance ~ C(identity) * C(alignment)', data=subset).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Extract p-values for interactions
            if 'C(identity):C(alignment)' in anova_table.index:
                p_val = anova_table.loc['C(identity):C(alignment)', 'PR(>F)']
            else:
                p_val = float('nan')  # If the term is not in the ANOVA table

            post_hoc_results.append({
                'Condition': cond,
                'Raw p-value': p_val
            })

            # If 2-way interaction is significant, perform post-hoc tests for each alignment
            if p_val <= 0.05:
                print(f"2-way interaction significant for condition {cond}. Performing further post-hoc tests.")
                
                alignments = subset['alignment'].unique()
                for align in alignments:
                    align_subset = subset[subset['alignment'] == align]
                    post_hoc_model = ols('distance ~ C(identity)', data=align_subset).fit()
                    post_hoc_anova = sm.stats.anova_lm(post_hoc_model, typ=2)
                    post_hoc_p_val = post_hoc_anova.loc['C(identity)', 'PR(>F)']
                    
                    post_hoc_results.append({
                        'Condition': f'{cond} - {align}',
                        'Raw p-value': post_hoc_p_val
                    })

        # Convert results to DataFrame
        post_hoc_df = pd.DataFrame(post_hoc_results)
        
        # Apply FDR correction
        p_values = post_hoc_df['Raw p-value']
        corrected_pvals = multipletests(p_values, method='fdr_bh')[1]
        post_hoc_df['FDR-corrected p-value'] = corrected_pvals

        # Print or return the results
        for index, row in post_hoc_df.iterrows():
            print(f'Condition: {row["Condition"]}, Raw p-value: {row["Raw p-value"]:.4f}, FDR-corrected p-value: {row["FDR-corrected p-value"]:.4f}')

        # Ensure the results path exists
        os.makedirs(self.results_path, exist_ok=True)
        output_file_name = f"post_hoc_results_{self.model_type}.csv"
        output_file_path = os.path.join(self.results_path, output_file_name)
        # Save results to CSV
        post_hoc_df.to_csv(output_file_path, index=False)
        print(f"Post hoc results saved to {output_file_name}")


    def complete_design(self):
        # Check lengths of each group
        print(f"Lengths - congruency: {len(self.df['congruency'])}, identity: {len(self.df['identity'])}, alignment: {len(self.df['alignment'])}")
            
        if len(self.df['congruency']) < 2 or len(self.df['identity']) < 2 or len(self.df['alignment']) < 2:
            raise ValueError("Insufficient data points in one or more factors for statistical analysis.")
            
        # Run three-way ANOVA
        model = ols('distance ~ C(congruency) * C(identity) * C(alignment)', data=self.df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
            
        # Print ANOVA table
        print(anova_table)
        
        # Plot bar plot
        means = self.df.groupby(['congruency', 'alignment', 'identity'])['distance'].mean().reset_index()
        g = sns.catplot(x='alignment', y='distance', hue='identity', col='congruency', data=means, kind='bar', palette='muted')
        g.set_ylabels('Mean Distance')
        g.fig.suptitle('Mean Distances for Different Conditions')
        plt.show()

        self.results = anova_table
        return anova_table
        
    def analyze(self, partial_design=True):
            if partial_design:
                self.partial_design()
            else:
                self.complete_design()

if __name__ == "__main__":
    pass