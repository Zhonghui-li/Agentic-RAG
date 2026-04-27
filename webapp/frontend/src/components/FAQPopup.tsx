// components/FAQPopup.tsx
import { Box, Typography, Modal } from '@mui/material';
import { X } from 'lucide-react';
import type React from 'react';

const modalStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 500,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

type FAQPopupProps = {
  open: boolean;
  handleClose: () => void;
};

const FAQPopup: React.FC<FAQPopupProps> = ({ open, handleClose }) => (
  <Modal
    aria-describedby="faq-modal-description"
    aria-labelledby="faq-modal-title"
    onClose={handleClose}
    open={open}
  >
    <Box sx={modalStyle}>
      <Box alignItems="center" display="flex" justifyContent="space-between">
        <Typography id="faq-modal-title" variant="h6">
          Updates & FAQ
        </Typography>
        <button onClick={handleClose}>
          <X />
        </button>
      </Box>
      <Typography id="faq-modal-description" sx={{ mt: 2 }}>
        The LLM Logic project is being developed by members of the AIEA Lab under the supervision of
        Professor Leilani Gilpin.
      </Typography>
    </Box>
  </Modal>
);

export default FAQPopup;
