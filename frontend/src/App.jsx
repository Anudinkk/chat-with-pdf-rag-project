import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Plus, Send, Mic, User, BotMessageSquare, LoaderCircle, FileCheck, X } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8002';

// --- Child Component (Moved outside App) ---
const ChatMessage = ({ sender, message }) => {
    const isUser = sender === 'user';
    // Ensure message is always a string or convert non-strings safely
    const displayMessage = (typeof message === 'string') ? message : JSON.stringify(message, null, 2);

    return (
        <div className="w-full max-w-3xl mx-auto">
            <div className="flex items-start gap-4 my-6">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${isUser ? 'bg-indigo-500' : 'bg-gray-700'}`}>
                    {isUser ? <User size={18} /> : <BotMessageSquare size={18} />}
                </div>
                <div className="flex-grow bg-gray-800 p-4 rounded-lg">
                    <p className="whitespace-pre-wrap leading-relaxed">{displayMessage}</p>
                </div>
            </div>
        </div>
    );
};


// --- Main App Component ---
const App = () => {
    // --- State Management ---
    const [file, setFile] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    // New state for status messages (uploading, success, error)
    const [statusMessage, setStatusMessage] = useState(null); // { text: string, type: 'error' | 'success' | 'loading' | 'info' }

    const fileInputRef = useRef(null);
    const chatEndRef = useRef(null);

    // --- Effects ---
    useEffect(() => {
        setSessionId(`session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`);
    }, []);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatHistory, isLoading]);

    // --- Event Handlers ---
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.type === 'application/pdf') {
            setStatusMessage(null); // Clear previous errors
            handleFileUpload(selectedFile);
        } else if (selectedFile) { // if a file was selected but it wasn't a PDF
            setStatusMessage({ text: 'Please select a valid PDF file.', type: 'error' });
            // Reset file input if invalid file selected
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        }
    };

    const handleFileUpload = async (fileToUpload) => {
        if (!sessionId) {
            setStatusMessage({ text: 'Session not initialized. Please refresh.', type: 'error' });
            return;
        }
        const formData = new FormData();
        formData.append('file', fileToUpload);
        formData.append('session_id', sessionId);

        setIsLoading(true); // For the "Thinking..." bubble
        setChatHistory([]);
        // Show "Uploading..." message in green, as requested
        setStatusMessage({ text: `Uploading "${fileToUpload.name}"...`, type: 'loading' });
        setFile(null); // Clear previous file state immediately

        try {
            await axios.post(`${API_BASE_URL}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setFile(fileToUpload); // Set file state only on success
            setChatHistory([{
                sender: 'ai',
                message: `Ready to answer questions about "${fileToUpload.name}".`
            }]);
            // Show success message in green
            setStatusMessage({ text: `Successfully uploaded "${fileToUpload.name}".`, type: 'success' });
        } catch (err) {
            console.error("File upload error details:", err.response || err);
            let specificError = 'An error occurred during file upload.';
            if (err.response && err.response.status === 422 && err.response.data?.detail) {
                try {
                    if (Array.isArray(err.response.data.detail)) {
                        specificError = err.response.data.detail.map(d => d.msg || JSON.stringify(d)).join(', ');
                    } else if (typeof err.response.data.detail === 'string') {
                        specificError = err.response.data.detail;
                    }
                } catch (parseError) {
                    console.error("Could not parse error detail:", parseError);
                }
            } else if (err.response?.data?.detail) {
                specificError = err.response.data.detail;
            }
            setStatusMessage({ text: specificError, type: 'error' });
            setFile(null); // Ensure file state is null on error
        } finally {
            setIsLoading(false);
            // Reset file input after upload attempt regardless of outcome
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        }
    };

    const handleSendMessage = async () => {
        if (!userInput.trim() || !file || isLoading) return;

        // Clear file status messages when sending a chat message
        setStatusMessage(null); 

        const newHistory = [...chatHistory, { sender: 'user', message: userInput }];
        setChatHistory(newHistory);
        const currentUserInput = userInput;
        setUserInput('');
        setIsLoading(true);

        try {
            const response = await axios.post(`${API_BASE_URL}/chat`, {
                session_id: sessionId,
                message: currentUserInput,
            });
            const answer = response?.data?.answer;
            if (typeof answer !== 'string') {
                console.error("Unexpected response format:", response.data);
                throw new Error("Received an unexpected response from the server.");
            }
            setChatHistory(prev => [...prev, { sender: 'ai', message: answer }]);
        } catch (err) {
            console.error("Chat error details:", err.response || err);
            const errorMsg = err.response?.data?.detail || err.message || 'An error occurred.';
            // Add error as an AI message in chat history
            setChatHistory(prev => [...prev, { sender: 'ai', message: `Sorry, an error occurred: ${errorMsg}` }]);
        } finally {
            setIsLoading(false);
        }
    };

    // New handler to remove the file and reset state
    const handleRemoveFile = () => {
        setFile(null);
        setChatHistory([]);
        setStatusMessage({ text: "File removed. Please upload a new PDF.", type: 'info' });
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
        // Reset session ID to start clean on the backend
        setSessionId(`session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`);
    };

    const hasChatStarted = chatHistory.length > 0;

    return (
        // The root div now uses h-full instead of h-screen
        <div key={sessionId} className="flex flex-col h-full bg-[#131314] text-gray-200 font-sans">
            <div className="flex-1 flex flex-col min-h-0">
                {/* Chat History */}
                <div className="flex-1 overflow-y-auto p-4">
                    {hasChatStarted && (
                        <div className="h-full">
                            {chatHistory.map((entry, index) => (
                                // Use a more robust key including session ID
                                <ChatMessage key={`${sessionId}-msg-${index}`} sender={entry.sender} message={entry.message} />
                            ))}
                            {isLoading && (
                                <div className="w-full max-w-3xl mx-auto">
                                    <div className="flex items-start gap-4 my-6">
                                        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
                                            <BotMessageSquare size={18} />
                                        </div>
                                        <div className="flex-grow bg-gray-800 p-4 rounded-lg flex items-center">
                                            <LoaderCircle className="animate-spin mr-3" />
                                            <span>Thinking...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>
                    )}
                </div>
            </div>

            {/* Input Area */}
            <div className={`w-full p-4 flex flex-col items-center shrink-0 ${hasChatStarted ? 'bg-transparent' : 'flex-grow justify-center'}`}>
                {!hasChatStarted && !isLoading && (
                    <h1 className="text-5xl font-semibold text-gray-500/80 mb-8 tracking-tight">
                        What's on the agenda today?
                    </h1>
                )}
                <div className="w-full max-w-3xl">

                    {/* --- Active File Chip --- */}
                    {file && (
                        <div className="mb-2 flex justify-center">
                            <div className="bg-gray-700/60 border border-gray-600/80 text-gray-300 text-sm rounded-full px-4 py-1.5 inline-flex items-center gap-2 shadow-md">
                                <FileCheck size={16} className="text-green-400" />
                                <span>Active: <strong>{file.name}</strong></span>
                                <button
                                    onClick={handleRemoveFile}
                                    className="ml-2 text-gray-400 hover:text-white transition-colors"
                                    title="Remove file"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        </div>
                    )}

                    {/* --- Upload/Error Status Message --- */}
                    {statusMessage && (
                        <div className="mb-3 rounded-lg p-3 flex items-center justify-center w-full">
                            <p className={`text-sm font-medium text-center ${
                                statusMessage.type === 'error' ? 'text-red-400' :
                                (statusMessage.type === 'success' || statusMessage.type === 'loading') ? 'text-green-400' : // Loading and Success are green
                                'text-gray-400' // for 'info'
                            }`}>
                                {statusMessage.text}
                            </p>
                        </div>
                    )}

                    <div className="bg-[#1E1F20] border border-gray-700/50 rounded-full p-2 flex items-center shadow-lg w-full transition-all duration-300 h-14">
                        <button
                            onClick={() => fileInputRef.current.click()}
                            className="p-2.5 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-gray-700"
                            title="Upload PDF"
                        >
                            <Plus size={20} />
                        </button>
                        <input
                            type="file"
                            accept=".pdf"
                            onChange={handleFileChange}
                            ref={fileInputRef}
                            className="hidden"
                        />
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                            placeholder={file ? `Ask about ${file.name}` : "Upload a PDF to ask anything"}
                            className="flex-grow bg-transparent text-white placeholder-gray-500 focus:outline-none px-4 text-md h-full"
                            disabled={!file || isLoading}
                        />
                        <button className="p-2.5 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-gray-700">
                            <Mic size={20} />
                        </button>
                        <button
                            onClick={handleSendMessage}
                            className="p-2.5 text-gray-400 hover:text-white transition-colors disabled:opacity-50 rounded-full hover:bg-gray-700"
                            disabled={!userInput.trim() || isLoading}
                        >
                            <Send size={20} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;