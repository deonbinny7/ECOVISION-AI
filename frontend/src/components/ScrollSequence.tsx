"use client";

import React, { useRef, useEffect, useState } from "react";
import { useScroll, useTransform, motion } from "framer-motion";

const FRAME_COUNT = 100;

export default function ScrollSequence() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [images, setImages] = useState<HTMLImageElement[]>([]);
  const [loaded, setLoaded] = useState(false);

  // Preload images
  useEffect(() => {
    const loadedImages: HTMLImageElement[] = [];
    let loadedCount = 0;

    for (let i = 0; i < FRAME_COUNT; i++) {
        const img = new Image();
        const frameStr = i.toString().padStart(4, "0");
        img.src = `/sequence/${frameStr}.webp`;
        
        img.onload = () => {
            loadedCount++;
            if (loadedCount === FRAME_COUNT) {
                setImages(loadedImages);
                setLoaded(true);
            }
        };
        img.onerror = () => {
             // Fallback if sequence is missing so we don't block forever
             loadedCount++;
             if (loadedCount === FRAME_COUNT) {
                setImages(loadedImages);
                setLoaded(true);
             }
        };
        loadedImages.push(img);
    }
  }, []);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  const frameIndex = useTransform(scrollYProgress, [0, 1], [0, Math.max(0, FRAME_COUNT - 1)]);

  useEffect(() => {
    if (!loaded || !canvasRef.current || images.length === 0) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    return frameIndex.on("change", (latest) => {
        const idx = Math.floor(latest);
        if (images[idx] && images[idx].complete) {
            ctx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height);
            // object-fit cover logic
            const canvasWidth = canvasRef.current!.width;
            const canvasHeight = canvasRef.current!.height;
            const imgRatio = images[idx].width / images[idx].height;
            const canvasRatio = canvasWidth / canvasHeight;

            let renderWidth = canvasWidth;
            let renderHeight = canvasHeight;
            let offsetX = 0;
            let offsetY = 0;

            if (imgRatio > canvasRatio) {
               renderWidth = canvasHeight * imgRatio;
               offsetX = (canvasWidth - renderWidth) / 2;
            } else {
               renderHeight = canvasWidth / imgRatio;
               offsetY = (canvasHeight - renderHeight) / 2;
            }

            ctx.drawImage(images[idx], offsetX, offsetY, renderWidth, renderHeight);
        }
    });
  }, [loaded, frameIndex, images]);

  // Initial draw
  useEffect(() => {
     if (loaded && canvasRef.current && images[0] && images[0].complete) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
             const canvas = canvasRef.current;
             ctx.drawImage(images[0], 0, 0, canvas.width, canvas.height); // simple fallback draw
          }
     }
  }, [loaded, images]);

  // Text overlay variants
  const section1Opacity = useTransform(scrollYProgress, [0, 0.15, 0.3], [1, 1, 0]);
  const section2Opacity = useTransform(scrollYProgress, [0.2, 0.4, 0.5], [0, 1, 0]);
  const section2X = useTransform(scrollYProgress, [0.2, 0.4], [-50, 0]);
  const section3Opacity = useTransform(scrollYProgress, [0.4, 0.6, 0.7], [0, 1, 0]);
  const section3X = useTransform(scrollYProgress, [0.4, 0.6], [50, 0]);
  const section4Opacity = useTransform(scrollYProgress, [0.6, 0.75, 0.85], [0, 1, 0]);
  const section5Opacity = useTransform(scrollYProgress, [0.8, 0.95, 1], [0, 1, 1]);
  const section5Y = useTransform(scrollYProgress, [0.8, 0.95], [50, 0]);
  const canvasOpacity = useTransform(scrollYProgress, [0.8, 1], [0.5, 0.1]);
  const scrollIndicatorOpacity = useTransform(scrollYProgress, [0, 0.1], [0.7, 0]);
  

  return (
    <div ref={containerRef} className="h-[250vh] relative w-full bg-[#0a0f1c]">
      <div className="sticky top-0 h-screen w-full overflow-hidden">
        {/* Subtle grid pattern for visual interest on plain black */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-30 pointer-events-none"></div>

        {/* Canvas */}
        <motion.canvas
          ref={canvasRef}
          width={1920}
          height={1080}
          style={{ opacity: canvasOpacity }}
          className="absolute inset-0 w-full h-full object-cover mix-blend-screen"
        />

        {/* Overlay Content */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none px-8">
            <motion.div style={{ opacity: section1Opacity }} className="absolute text-center">
                <h1 className="text-4xl md:text-6xl font-bold tracking-tight">Waste management is one of the largest global challenges</h1>
            </motion.div>

            <motion.div style={{ opacity: section2Opacity, x: section2X }} className="absolute text-left max-w-2xl md:left-[10%]">
                <h2 className="text-3xl md:text-5xl font-semibold opacity-90">Manual sorting is inefficient, inconsistent, and unsustainable</h2>
            </motion.div>

            <motion.div style={{ opacity: section3Opacity, x: section3X }} className="absolute text-right max-w-2xl md:right-[10%]">
                <h2 className="text-3xl md:text-5xl font-semibold text-cyan-400">Deep learning can extract patterns and automate classification</h2>
            </motion.div>

            <motion.div style={{ opacity: section4Opacity }} className="absolute text-center">
                <h2 className="text-4xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-cyan-500">Introducing EcoVision AI</h2>
            </motion.div>

            <motion.div style={{ opacity: section5Opacity, y: section5Y }} className="absolute text-center">
                <h2 className="text-5xl md:text-8xl font-black">Explainable<br/><span className="text-green-400">Waste Intelligence</span></h2>
            </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div style={{ opacity: scrollIndicatorOpacity }} className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center animate-bounce">
            <p className="text-sm font-medium uppercase tracking-widest mb-2 text-cyan-400">Scroll to explore</p>
            <div className="w-px h-16 bg-gradient-to-b from-cyan-400 to-transparent"></div>
        </motion.div>
      </div>
    </div>
  );
}
