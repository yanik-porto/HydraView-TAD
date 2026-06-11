import json

class TextFromJson:
    def __init__(self, json_file, specific_index=-1, training=True):
        with open(json_file, 'r') as f:
            textmap = json.load(f)
        self.specific_index = specific_index
        self.training = training

        textlist = []
        for label_vals in textmap.values():
            viewlist = []
            for view_vals in label_vals.values():
                viewlist.append(view_vals)
            textlist.append(viewlist)
        self.textlist = textlist

    def __call__(self, results):
        # keypoints = results['keypoint']

        # assert len(labels) == 1

        if self.training:
            label = results['label']
            viewlist = self.textlist[label]


            texts = []
            # for iv in range(len(keypoints)):
            if self.specific_index == -1:
                assert 'indexes' in results, results.keys() 
                texts.append(viewlist[results['indexes'][0]]) #for now, as long as no multiview training with specified index
            else:
                texts.append(viewlist[self.specific_index])

        else:
            texts = []
            if self.specific_index >= 0:
                for viewlist in self.textlist:
                    texts.append(viewlist[self.specific_index])
            elif self.specific_index == -2:
                assert 'indexes' in results, results.keys()
                for viewlist in self.textlist:
                    texts.append(viewlist[results['indexes'][0]]) #for now, as long as no multiview training  with specified index
            elif self.specific_index == -3:
                assert 'closest_node' in results, results.keys()
                assert len(results['closest_node']) > 0
                # print(results['closest_node'][0])
                for viewlist in self.textlist:
                    # print(len(viewlist))
                    texts.append(viewlist[results['closest_node'][0]]) #for now, as long as no multiview training  with specified index
            elif self.specific_index == -4:
                assert 'frame_dir' in results, results.keys()
                frame_dir = results["frame_dir"]
                r = int(frame_dir[13:16])
                assert r in (1, 2)
                index = 5 if r == 1 else 7
                for viewlist in self.textlist:
                    texts.append(viewlist[index])
            else:
                # for viewlist in self.textlist:
                texts = [list(row) for row in zip(*self.textlist)]


        results['texts'] = texts

        return results