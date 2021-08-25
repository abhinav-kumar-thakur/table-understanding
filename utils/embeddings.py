import os
import shutil

import numpy as np
from matplotlib import pyplot as plt, cm


def compare_embeddings(model, file_one, file_two, sheet_one, sheet_two):
    sheet_one_embeddings = model.predict_labels(file_one, [sheet_one])[sheet_one]['embeddings'].numpy()
    sheet_two_embeddings = model.predict_labels(file_two, [sheet_two])[sheet_two]['embeddings'].numpy()
    assert sheet_one_embeddings.shape == sheet_two_embeddings.shape

    row_count, column_count, _ = sheet_one_embeddings.shape
    distances = np.empty(shape=(row_count, column_count))
    for i in range(0, row_count):
        for j in range(0, column_count):
            distances[i][j] = np.linalg.norm(sheet_two_embeddings[i][j] - sheet_one_embeddings[i][j])

    flatten_distance = distances.flatten()
    max_distance = max(flatten_distance)
    median_distance = np.median(flatten_distance)
    avg_distance = np.average(flatten_distance)

    counts, bins = np.histogram(flatten_distance, bins=100)
    plt.title(f'{sheet_one} vs {sheet_two} - Max: {max_distance} Median: {median_distance} Avg: {avg_distance}')
    plt.hist(bins[:-1], bins, weights=counts)
    plt.plot()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    X = np.arange(1, column_count + 1, 1)
    Y = np.arange(1, row_count + 1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = distances
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, max_distance + 1)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def compare_embeddings_when_transposed(model, file_one, file_two, sheet_one, sheet_two):
    sheet_one_embeddings = model.predict_labels(file_one, [sheet_one])[sheet_one]['embeddings'].numpy()
    sheet_two_embeddings = model.predict_labels(file_two, [sheet_two])[sheet_two]['embeddings'].numpy()
    assert sheet_one_embeddings.shape[0] == sheet_two_embeddings.shape[1]
    assert sheet_one_embeddings.shape[1] == sheet_two_embeddings.shape[0]

    row_count, column_count, _ = sheet_one_embeddings.shape
    distances = np.empty(shape=(row_count, column_count))
    for i in range(0, row_count):
        for j in range(0, column_count):
            distances[i][j] = np.linalg.norm(sheet_two_embeddings[j][i] - sheet_one_embeddings[i][j])

    flatten_distance = distances.flatten()
    max_distance = max(flatten_distance)
    median_distance = np.median(flatten_distance)
    avg_distance = np.average(flatten_distance)

    counts, bins = np.histogram(flatten_distance, bins=100)
    plt.title(f'{sheet_one} vs {sheet_two} - Max: {max_distance} Median: {median_distance} Avg: {avg_distance}')
    plt.hist(bins[:-1], bins, weights=counts)
    plt.plot()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    X = np.arange(1, column_count + 1, 1)
    Y = np.arange(1, row_count + 1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = distances
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, max_distance + 1)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def generate_visualisation_files(model, filename, sheet_name):
    print('Saving sheet data')
    embeddings = model.predict_labels(filename, [sheet_name])
    embeddings_output = f'../output/sheet/{sheet_name}'
    if os.path.exists(embeddings_output):
        shutil.rmtree(embeddings_output)
    metadata = [['text', 'label', 'coordinates']]
    cell_embeddings = []
    for i in range(0, len(embeddings[sheet_name]['table_arrays'])):
        for j in range(0, len(embeddings[sheet_name]['table_arrays'][i])):
            value = embeddings[sheet_name]['table_arrays'][i][j]

            cell_embeddings.append(list(embeddings[sheet_name]['embeddings'][i][j]))

            text = ''
            if value != 'None' and value != '':
                text = value.replace('\n', '')

            metadata.append([text, embeddings[sheet_name]['labels'][i][j], f'|{i + 1}_{j + 1}|'])

            assert len(cell_embeddings) + 1 == len(metadata)
    print('Saving:', sheet_name)
    os.makedirs(embeddings_output)
    np.savetxt(os.path.join(embeddings_output, 'embeddings.tsv'), cell_embeddings, delimiter='\t')
    np.savetxt(os.path.join(embeddings_output, 'metadata.tsv'), metadata, delimiter='\t', fmt='%s')
    print('done')


def generate_sheet_pair_embedding_visualisation_files(model, file_and_sheets):
    print('Saving sheet pair data')
    metadata = [['sheet', 'text', 'label', 'coordinates']]
    embeddings_output = f'../output/sheet_pair/{"(&)".join([s for f, s in file_and_sheets])}'
    cell_embeddings = []
    for filename, sheet_name in file_and_sheets:
        embeddings = model.predict_labels(filename, [sheet_name])
        if os.path.exists(embeddings_output):
            shutil.rmtree(embeddings_output)

        for i in range(0, len(embeddings[sheet_name]['table_arrays'])):
            for j in range(0, len(embeddings[sheet_name]['table_arrays'][i])):
                value = embeddings[sheet_name]['table_arrays'][i][j]
                cell_embeddings.append(list(embeddings[sheet_name]['embeddings'][i][j]))

                text = ''
                if value != 'None' and value != '':
                    text = value.replace('\n', '')

                metadata.append([sheet_name, text, embeddings[sheet_name]['labels'][i][j], f'|{i + 1}_{j + 1}|'])

                assert len(cell_embeddings) + 1 == len(metadata)

    print('Saving:', embeddings_output)
    os.makedirs(embeddings_output)
    np.savetxt(os.path.join(embeddings_output, 'embeddings.tsv'), cell_embeddings, delimiter='\t')
    np.savetxt(os.path.join(embeddings_output, 'metadata.tsv'), metadata, delimiter='\t', fmt='%s')
    print('done')