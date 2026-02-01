import React, { useState } from "react";
import { Play } from "lucide-react";

interface PredictionResult {
  predictedClass: string;
  confidence: number;
  model: string;
} 
import Header from "@/components/Header";
import ImageUpload from "@/components/ImageUpload";
import ClassificationResult from "@/components/ClassificationResult";
import ExplanationSection from "@/components/ExplanationSection";
import Footer from "@/components/Footer";

const Index: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handleClassify = () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setResult(null);

    // Simulation d'une classification (à remplacer par votre API Python)
    setTimeout(() => {
      const classes = ["Chien", "Chat", "Oiseau", "Voiture", "Avion", "Chiffre 7", "Lettre A"];
      const models = ["KNN (K=3)", "SVM (RBF)", "Random Forest"];
      
      setResult({
        predictedClass: classes[Math.floor(Math.random() * classes.length)],
        confidence: Math.floor(Math.random() * 20) + 80,
        model: models[Math.floor(Math.random() * models.length)],
      });
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6">
        <Header />

        <main className="space-y-8">
          {/* Upload Section */}
          <section className="bg-card rounded-2xl p-6 md:p-8 border border-border shadow-sm">
            <h2 className="text-xl font-semibold text-foreground mb-6">
              Tester le modèle
            </h2>
            
            <div className="space-y-6">
              <ImageUpload
                onImageSelect={setSelectedImage}
                selectedImage={selectedImage}
              />

              <div className="flex justify-center">
                <button
                  type="button"
                  onClick={handleClassify}
                  disabled={!selectedImage || isLoading}
                  aria-busy={isLoading}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg disabled:opacity-50"
                >
                  <Play className="w-5 h-5" />
                  Lancer la classification
                </button>
              </div>

              <ClassificationResult result={result} isLoading={isLoading} />
            </div>
          </section>

          {/* Explanation Section */}
          <ExplanationSection />
        </main>

        <Footer />
      </div>
    </div>
  );
};

export default Index;
