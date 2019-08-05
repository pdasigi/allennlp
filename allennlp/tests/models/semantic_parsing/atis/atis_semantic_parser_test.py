from allennlp.common.testing import ModelTestCase


class AtisSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(AtisSemanticParserTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / "semantic_parsing" / "atis" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "atis" / "sample.json")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
