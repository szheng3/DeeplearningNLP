import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.examples.nlp.word2vec.Word2VecRawTextExample;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

public class naturallearningprocessing {
	public static void main(String[] args) throws Exception {
		System.out.println("sdf");

		String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

		System.out.println("Load Sentences....");
		SentenceIterator iter = new BasicLineIterator(filePath);
		TokenizerFactory t = new DefaultTokenizerFactory();

		t.setTokenPreProcessor(new CommonPreprocessor());
		System.out.println("Building ....");

		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(1).layerSize(100).seed(42).windowSize(5)
				.iterate(iter).tokenizerFactory(t).build();
		System.out.println("Fitting Word2Vec model....");

		vec.fit();
		System.out.println("Writing word vectors to text file....");

		WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

		System.out.println("Closest Words:");
		Collection<String> lst = vec.wordsNearest("night", 10);
		System.out.println("10 Words closest to 'night': " + lst);

		UiServer server = UiServer.getInstance();
		System.out.println("Started on port " + server.getPort());
	}
}
