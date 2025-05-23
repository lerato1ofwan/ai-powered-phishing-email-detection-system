'use client';
import Image from "next/image";
import React, { useEffect, useState } from 'react';

interface PredictionResponse {
  prediction: string;
  label: number; // 0 for Legitimate, 1 for Phishing
  confidence: number;
  explanation: [string, number][]; // Array of [word, weight] tuples
  error?: string | null; // Optional error message
}

export default function Home() {
  const [subject, setSubject] = useState('');
  const [sender, setSender] = useState('');
  const [body, setBody] = useState('');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("nb"); // Default to MultinomialNB

  // Check API endpoint was correct setup
  useEffect(() => {
    if (!process.env.NEXT_PUBLIC_API_ENDPOINT) {
      console.error('API endpoint not correctly configured');
    }
  }, []);

  const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT;
  if (!apiEndpoint) {
    throw new Error('NEXT_PUBLIC_API_ENDPOINT environment variable is not defined');
  }    

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setResult(null);
    setApiError(null);

    try {
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, sender, body, model_choice: selectedModel }),
      });
      const data: PredictionResponse = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || `API Error: ${response.statusText}`);
      }
      setResult(data);
      setIsModalOpen(true); // Open the modal on successful result
    } catch (error: any) {
      setApiError(error.message || "Failed to fetch prediction. Check API status and console.");
      setIsModalOpen(true); // Also open the modal on error
    } finally {
      setIsLoading(false);
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setResult(null); // Clear result on close
    setApiError(null); // Clear API error on close
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const allowedTypes = ['text/plain', 'text/csv'];
      const fileExtension = file.name.split('.').pop()?.toLowerCase();

      if (allowedTypes.includes(file.type) || (fileExtension === 'txt' || fileExtension === 'csv')) {
        setSelectedFile(file);
        setApiError(null); // Clear any previous file errors
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target?.result as string;
          setBody(text); // Populate body textarea with file content
        };
        reader.onerror = () => {
          console.error("Error reading file");
          setApiError("Could not read the selected file.");
          setIsModalOpen(true);
          setSelectedFile(null);
        }
        reader.readAsText(file);
      } else {
        setApiError(`Invalid file type. Please upload a .txt or .csv file. You uploaded a .${fileExtension} file.`);
        setIsModalOpen(true);
        setSelectedFile(null);
        // Reset the file input
        if (event.target) {
          event.target.value = '';
        }
      }
    } else {
      setSelectedFile(null);
    }
  };

  const getExplanationClasses = (weight: number, predictedLabel: number): string => {
    const significanceThreshold = 0.01;
    let classes = "text-gray-700"; // Default for less significant or neutral words
    if (weight > significanceThreshold) {
      classes = predictedLabel === 1 ? "text-red-600 font-semibold" : "text-green-600 font-semibold";
    }
    return classes;
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] justify-items-center min-h-screen pt-4 px-8 pb-8 gap-4 sm:px-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <h3 className="w-full text-xl text-center font-semibold">AI-Powered Phishing Email Detection System</h3>
        <div className="w-full text-center space-y-3">
          <p className="text-sm/6 font-[family-name:var(--font-geist-mono)] tracking-[-.01em]">
            Get started by selecting a model:
          </p>
          <div className="inline-block relative">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="block appearance-none w-auto bg-white border border-gray-300 hover:border-gray-400 px-4 py-2 pr-8 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm text-gray-900"
            >
              <option value="nb">MultinomialNB</option>
              <option value="bert-mini">BERT-mini</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
            </div>
          </div>
        </div>

        {/* --- START OF AI PHISHING DETECTOR --- */}
        <div className="w-full max-w-2xl font-[family-name:var(--font-geist-sans)]">
          <div className="mb-8 md:flex md:items-start md:gap-6">

            <div className="mb-6 md:mb-0 text-center md:flex-[3_1_0%]">
              <p className="text-md text-gray-600 text-center">
                Enter email details below to analyze for potential phishing threats.
              </p>
            </div>

            <div className="mb-6 md:mb-0 text-center md:flex-[1_1_0%]"> 
              <p className="text-md text-gray-600 text-center">
                Or
              </p>
            </div>

            {/* File Upload Section (Moved from form) */}
            <div className="text-center md:flex-[3_1_0%]" >
              <div className="flex flex-col items-center justify-center w-full">
                <label htmlFor="emailFile" className="block text-sm font-medium text-gray-700 mb-1.5">
                  Upload email file (.txt or .csv)
                </label>
                <div className="block flex flex-col items-center"> 
                  <input
                    type="file"
                    id="emailFile"
                    accept=".txt,.csv"
                    onChange={handleFileChange}
                    className="block text-sm text-gray-500
        file:mr-4 file:py-2 file:px-3
        file:rounded-md file:border-0
        file:text-sm file:font-semibold
        file:bg-indigo-50 file:text-indigo-700
        hover:file:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  />
                  {selectedFile && (
                    <div className="mt-2 text-xs text-gray-600 text-center"> {/* MODIFIED: Added text-center here */}
                      Selected: {selectedFile.name}
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedFile(null);
                          setSender('');
                          setSubject('');
                          setBody('');
                          const fileInput = document.getElementById('emailFile') as HTMLInputElement;
                          if (fileInput) fileInput.value = ''; 
                        }}
                        className="ml-2 text-indigo-600 hover:text-indigo-800 font-medium"
                      >
                        Clear
                      </button>
                    </div>
                  )}
                </div></div>

            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 sm:p-8 rounded-lg shadow-md">
            <div>
              <label htmlFor="sender" className="block text-sm font-medium text-gray-700 mb-1.5">Sender Email</label>
              <input
                type="text"
                id="sender"
                value={sender}
                onChange={(e) => setSender(e.target.value)}
                placeholder="e.g., sender@example.com"
                className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
            <div>
              <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-1.5">Subject</label>
              <input
                type="text"
                id="subject"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g., Urgent Account Update"
                className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
            <div>
              <label htmlFor="body" className="block text-sm font-medium text-gray-700 mb-1.5">Email Body</label>
              <textarea
                id="body"
                rows={10}
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Paste the full email body here..."
                required
                className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
            <button
              type="submit"
              disabled={isLoading || !body}
              className="w-full flex items-center justify-center rounded-md border border-transparent bg-foreground text-background px-6 py-3 text-base font-medium shadow-sm hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Analyzing...' : 'Check Email'}
            </button>
          </form>

          {isLoading && (
            <div className="text-center py-6 mt-6">
              <p className="text-lg text-gray-600 animate-pulse">Checking email, please wait...</p>
            </div>
          )}

        </div>
        {/* --- END OF AI PHISHING DETECTOR --- */}

      </main>
      <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://huggingface.co/lleratodev"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/hf_mini_logo.svg"
            alt="Hugging Face Icon"
            width={16}
            height={16}
          />
          Access the model on hugging face →
        </a>
      </footer>

      {/* --- RESULTS MODAL --- */}
      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 p-4 font-[family-name:var(--font-geist-sans)]"
          onClick={closeModal} // Close modal on overlay click
        >
          <div
            className="bg-white rounded-lg shadow-xl w-full max-w-xl max-h-[90vh] overflow-y-auto p-6 sm:p-8 space-y-6"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the modal content
          >
            {/* Modal Content: Success or Error */}
            {result && !result.error && (
              <>
                <div className="flex justify-between items-center">
                  <h2 className="text-2xl font-semibold text-gray-800">Analysis Result</h2>
                  <button
                    onClick={closeModal}
                    className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
                    aria-label="Close modal"
                  >
                    &times;
                  </button>
                </div>
                <div className="text-center">
                  <p className="text-base font-medium text-gray-600">Prediction</p>
                  <p className={`text-3xl font-bold ${result.label === 1 ? 'text-red-600' : 'text-green-600'}`}>
                    {result.prediction}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-base font-medium text-gray-600">Confidence</p>
                  <p className="text-2xl text-gray-800">
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                </div>
                {result.explanation && result.explanation.length > 0 && result.explanation[0][0] !== "LIME explanation error or N/A" && (
                  <div className="pt-6 border-t border-gray-200">
                    <h3 className="text-xl font-semibold text-gray-800 mb-3 text-center">Key Factors</h3>
                    <p className="text-sm text-gray-600 text-center mb-4">
                      Top words influencing the "{result.prediction}" classification (positive weights support this prediction):
                    </p>
                    <ul className="space-y-1.5 max-w-md mx-auto text-sm text-left bg-gray-50 p-4 rounded-md font-[family-name:var(--font-geist-mono)]">
                      {result.explanation.map(([word, weight]) => (
                        <li key={word + weight} className={getExplanationClasses(weight, result.label)}>
                          <span className="break-all">"{word}"</span>: {weight.toFixed(4)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.explanation && result.explanation.length > 0 && result.explanation[0][0] === "LIME explanation error or N/A" && (
                  <p className="text-center text-gray-600 pt-6 border-t border-gray-200">Could not generate detailed explanation for this email.</p>
                )}
              </>
            )}
            {apiError && (
              <div className="text-center">
                <h2 className="text-2xl font-semibold text-red-700 mb-4">Analysis Failed</h2>
                <p className="text-md text-red-600">{apiError}</p>
                <button
                  onClick={closeModal}
                  className="mt-6 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
                >
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      {/* --- END OF RESULTS MODAL --- */}
    </div>
  )};