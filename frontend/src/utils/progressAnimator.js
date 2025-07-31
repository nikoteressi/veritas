/**
 * Progress Animation Controller
 * 
 * Provides smooth progress animations using requestAnimationFrame
 * with customizable easing functions and duration control.
 */

class ProgressAnimator {
    constructor(options = {}) {
        this.duration = options.duration || 300; // Default 300ms
        this.easing = options.easing || this.easeOutCubic;
        this.updateInterval = options.updateInterval || 16; // ~60fps
        this.onUpdate = options.onUpdate || (() => {});
        this.onComplete = options.onComplete || (() => {});
        
        this.isAnimating = false;
        this.animationId = null;
        this.startTime = null;
        this.startValue = 0;
        this.targetValue = 0;
        this.currentValue = 0;
    }

    /**
     * Start animating from current value to target value
     * @param {number} targetValue - Target progress value (0-100)
     * @param {number} duration - Optional duration override
     */
    animateTo(targetValue, duration = null) {
        // Cancel any existing animation
        this.stop();
        
        this.startValue = this.currentValue;
        this.targetValue = Math.max(0, Math.min(100, targetValue));
        this.duration = duration || this.duration;
        this.startTime = performance.now();
        this.isAnimating = true;
        
        this._animate();
    }

    /**
     * Set progress immediately without animation
     * @param {number} value - Progress value (0-100)
     */
    setImmediate(value) {
        this.stop();
        this.currentValue = Math.max(0, Math.min(100, value));
        this.startValue = this.currentValue;
        this.targetValue = this.currentValue;
        this.onUpdate(this.currentValue);
    }

    /**
     * Stop current animation
     */
    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.isAnimating = false;
    }

    /**
     * Get current progress value
     * @returns {number} Current progress (0-100)
     */
    getCurrentValue() {
        return this.currentValue;
    }

    /**
     * Check if animation is currently running
     * @returns {boolean} Animation status
     */
    isRunning() {
        return this.isAnimating;
    }

    /**
     * Internal animation loop
     * @private
     */
    _animate() {
        if (!this.isAnimating) return;

        const currentTime = performance.now();
        const elapsed = currentTime - this.startTime;
        const progress = Math.min(elapsed / this.duration, 1);

        // Apply easing function
        const easedProgress = this.easing(progress);
        
        // Calculate current value
        this.currentValue = this.startValue + (this.targetValue - this.startValue) * easedProgress;
        
        // Update callback
        this.onUpdate(this.currentValue);

        // Check if animation is complete
        if (progress >= 1) {
            this.currentValue = this.targetValue;
            this.isAnimating = false;
            this.onComplete(this.currentValue);
            return;
        }

        // Continue animation
        this.animationId = requestAnimationFrame(() => this._animate());
    }

    // Easing functions
    easeLinear(t) {
        return t;
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    easeOutQuart(t) {
        return 1 - Math.pow(1 - t, 4);
    }

    easeInOutQuart(t) {
        return t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
    }

    easeOutExpo(t) {
        return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }

    easeOutBack(t) {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }
}

/**
 * Factory function to create progress animator with predefined configurations
 */
export const createProgressAnimator = {
    /**
     * Create a smooth animator for general progress updates
     */
    smooth: (onUpdate, onComplete) => new ProgressAnimator({
        duration: 300,
        easing: function(t) { return this.easeOutCubic(t); }.bind(new ProgressAnimator()),
        onUpdate,
        onComplete
    }),

    /**
     * Create a fast animator for quick updates
     */
    fast: (onUpdate, onComplete) => new ProgressAnimator({
        duration: 150,
        easing: function(t) { return this.easeOutQuart(t); }.bind(new ProgressAnimator()),
        onUpdate,
        onComplete
    }),

    /**
     * Create a slow animator for important transitions
     */
    slow: (onUpdate, onComplete) => new ProgressAnimator({
        duration: 600,
        easing: function(t) { return this.easeInOutCubic(t); }.bind(new ProgressAnimator()),
        onUpdate,
        onComplete
    }),

    /**
     * Create a bouncy animator for completion effects
     */
    bouncy: (onUpdate, onComplete) => new ProgressAnimator({
        duration: 400,
        easing: function(t) { return this.easeOutBack(t); }.bind(new ProgressAnimator()),
        onUpdate,
        onComplete
    })
};

export default ProgressAnimator;