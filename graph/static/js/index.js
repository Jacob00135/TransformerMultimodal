(() => {
    'use strict';

    const adjustLightInput = document.getElementById('adjust-light');
    adjustLightInput.addEventListener('input', adjustLight);
    adjustLight({target: {value: adjustLightInput.value}});

    function adjustLight(e) {
        const a = parseFloat((e.target.value * 0.01).toFixed(2));
        document.querySelector('body > .shade').style.backgroundColor = `rgba(0, 0, 0, ${a})`;
    }
})();
